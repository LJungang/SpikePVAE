# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os 
from models import ModifiedPoissonVAE 
from spikedataset import SpikeDataset, get_dataloaders 
from config import Config 
from torch.utils.tensorboard import SummaryWriter
from base.utils_model import *
import collections
from tqdm.autonotebook import tqdm 
import piqa 

class ModifiedTrainerVAE(object): 
    def __init__(self, model, cfg, device='cuda'):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optim = optim.Adamax(self.model.parameters(), lr=cfg.lr) 
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
            self.optim,
            T_max=self.cfg.epochs,  
            eta_min=1e-5,  
        )
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.project)) 
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        self.ssim = piqa.SSIM(n_channels=1).to(self.device) 
        self.stats = collections.defaultdict(dict) # 用于保存训练信息
        self.best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
        self.model.chkpt_dir = os.path.join(cfg.log_dir, cfg.project)
        os.makedirs(self.model.chkpt_dir, exist_ok=True)
        
    def iteration(self, epoch: int = 0):
        self.model.train()
        nelbo = AverageMeter()
        grads = AverageMeter()
        perdim_kl = AverageMeter()
        perdim_mse = AverageMeter()
        ssim_meter = AverageMeter() 

        batch_pbar = tqdm(
            iterable=self.train_loader,
            desc=f"Epoch {epoch+1}/{self.cfg.epochs} - Batch",
            leave=False, 
            dynamic_ncols=True,
        )

        for i, (spike_data, x) in enumerate(batch_pbar):  
            gstep = epoch * len(self.train_loader) + i
            if self.cfg.use_warmup and epoch < self.cfg.warmup_epochs: 
                lr = (self.cfg.lr * gstep / self.cfg.warmup_epochs / len(self.train_loader))
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                    
            x = x.to(self.device)
            spike_data = spike_data.to(self.device)
            self.optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                y, z_pred, dist, log_dr = self.model(x, spike_data)
                recon_loss = self.model.loss_recon(y, x).mean()
                kl_loss = self.model.loss_kl(log_dr).sum(dim=-1).mean()
                loss = recon_loss + self.cfg.beta * kl_loss
                ssim_val = self.ssim(y, x).mean().item()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)

            if self.cfg.grad_clip is not None:
                if epoch < self.cfg.warmup_epochs:
                    max_norm = self.cfg.grad_clip * 3
                else:
                    max_norm = self.cfg.grad_clip
                grad_norm = nn.utils.clip_grad_norm_(
                    parameters=self.parameters(),
                    max_norm=max_norm,
                ).item()
                grads.update(grad_norm)
                # self.stats['grad'][gstep] = grad_norm # 可以选择记录
                # if grad_norm > self.cfg.grad_clip:
                #     self.stats['loss'][gstep] = loss.item()

            with torch.inference_mode():
                nelbo.update((recon_loss + kl_loss).item())
                perdim_mse.update(recon_loss.item() / self.model.cfg.input_sz ** 2)
                perdim_kl.update(torch.mean(self.model.loss_kl(log_dr), dim=0).mean().item())
                ssim_meter.update(ssim_val)

            self.scaler.step(self.optim)
            self.scaler.update()
            #self.update_ema() #

            cond_schedule = (epoch >= self.cfg.warmup_epochs and self.scheduler is not None)
            if cond_schedule:
                self.scheduler.step()

            # 记录统计量 (可选)
            current_lr = self.optim.param_groups[0]['lr']
            # self.stats['lr'][gstep] = current_lr

            # 写入 TensorBoard (可选)
            cond_write = (gstep > 0 and self.writer is not None and gstep % self.cfg.log_freq == 0)
            if not cond_write:
                continue
            to_write = {
                'coeffs/beta': self.cfg.beta,
                # 'coeffs/temp': self.temperatures[gstep], # 不需要温度
                'coeffs/lr': self.optim.param_groups[0]['lr'],
            }
            to_write.update({
                **to_write,
                'train/loss_kl': kl_loss.item(),
                'train/loss_mse': recon_loss.item(),
                'train/nelbo_avg': nelbo.avg,
                'train/perdim_kl': perdim_kl.avg,
                'train/perdim_mse': perdim_mse.avg,
                'train/ssim': ssim_meter.avg, 
            })

            if self.cfg.grad_clip is not None:
                to_write['train/grad_norm'] = grads.avg

            # 写入 tensorboard
            for k, v in to_write.items():
                self.writer.add_scalar(k, v, gstep)

            # 重置
            if gstep % 100 == 0:
                grads.reset()
                nelbo.reset()
            # 更新 batch 进度条   <--- 修改这里
            batch_pbar.set_postfix(
                loss=f"{nelbo.avg:.4f}",
                kl=f"{kl_loss.item():.4f}",
                mse=f"{recon_loss.item():.4f}",
                ssim=f"{ssim_val:.4f}"
            )
        batch_pbar.close() # 关闭 batch 级别的进度条
        return nelbo.avg

    def train(self, epochs):
        # 添加一个 epoch 级别的进度条
        epoch_pbar = tqdm(range(epochs), desc="Training", position=0, leave=True, dynamic_ncols=True)
        for epoch in epoch_pbar:
            avg_loss = self.iteration(epoch)
            # print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            # 更新 epoch 级别的进度条
            epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")

            # 在这里可以添加验证和模型保存逻辑
            # if (epoch + 1) % self.cfg.chkpt_freq == 0: # 不需要根据 epoch 频率来保存
            #   self.validate(epoch=epoch)
            #   self.save(path=self.model.chkpt_dir, checkpoint=epoch+1)
            val_loss = self.validate(epoch=epoch)  # <--- 新增：调用 validate 方法
            if val_loss < self.best_val_loss:     # <--- 新增：比较验证损失
                self.best_val_loss = val_loss      # <--- 新增：更新最佳损失
                self.save(path=self.model.chkpt_dir, checkpoint=epoch+1) # <--- 新增：保存模型
                print(f"Epoch {epoch+1}: Best validation loss: {val_loss:.4f} - Saving model...")
        self.writer.close()
        epoch_pbar.close() # 关闭 epoch 级别的进度条

    @torch.inference_mode()
    def validate(self, epoch: int = -1):
      self.model.eval()
      nelbo = AverageMeter()
      for i, (spike_data, x) in enumerate(self.val_loader):
        # 数据移动到设备
        x = x.to(self.device)
        spike_data = spike_data.to(self.device)
        # 前向传播
        y, z_pred, dist, log_dr = self.model(x, spike_data)
        # 计算损失
        recon_loss = self.model.loss_recon(y, x).mean()
        kl_loss = self.model.loss_kl(log_dr).sum(dim=-1).mean()
        loss = recon_loss + self.cfg.beta * kl_loss
        nelbo.update(loss.item())
      # 记录到 tensorboard
      self.writer.add_scalar('val/nelbo_avg', nelbo.avg, epoch)
      print(f'Validation Loss: {nelbo.avg}')
      return nelbo.avg # 返回平均损失

    def parameters(self, requires_grad: bool = True):
      if requires_grad:
        return filter(
          lambda p: p.requires_grad,
          self.model.parameters(),
      )
      else:
        return self.model.parameters()
  
    def save(self, path: str, checkpoint: int = None):
        if checkpoint is not None:
            global_step = checkpoint * len(self.train_loader)
        else:
            global_step = None
        state_dict = {
            'metadata': {
                'timestamp': self.model.timestamp,
                'checkpoint': checkpoint,
                'global_step': global_step,
                'stats': self.stats,
                'root': path},
            'model': self.model.state_dict(),
            # 'model_ema': self.model_ema.state_dict()
            # if self.model_ema is not None else None,
            'optim': self.optim.state_dict(),
            'scaler': self.scaler.state_dict(),
            # 'scheduler': self.optim_schedule.state_dict()
            # if self.optim_schedule is not None else None,
        }
        fname = '+'.join([
            type(self.model).__name__,
            type(self).__name__],
        )
        if checkpoint is not None:
            fname = '-'.join([
                fname,
                f"{checkpoint:04d}"
            ])
        fname = f"{fname}_({now(True)}).pt"
        fname = pjoin(path, fname)
        torch.save(state_dict, fname)
        return

if __name__ == '__main__':
    # --- 1. 创建配置对象 ---
    # 假设你有一个 Config 类来保存配置 (config.py)
    cfg = Config()

    # --- 2. 实例化模型 ---
    model = ModifiedPoissonVAE(cfg)

    # --- 3. 实例化训练器 ---
    trainer = ModifiedTrainerVAE(model, cfg)

    # --- 4. 准备数据加载器 ---
    train_loader, val_loader = get_dataloaders(
        img_path=cfg.img_path,
        spike_path=cfg.spike_path,
        batch_size=cfg.batch_size,
        input_sz=cfg.input_sz,
        neurons_nums = cfg.n_latents,
        spike_times = cfg.spike_times
    )
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader
    # --- 5. 开始训练 ---
    trainer.train(epochs=cfg.epochs)