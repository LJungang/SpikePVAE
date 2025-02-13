# config.py
class Config:
    def __init__(self):
        # 数据集路径
        self.img_path = '/ssd-sata1/wj/dataset/celebaMask/img'
        self.spike_path = '/ssd-sata1/wj/dataset/celebaMask/celebaMask.npz'
        self.project = 'rnn+cnn_all_spike'
        # 模型参数
        self.neurons_nums = 100
        self.spike_times = 100
        self.n_latents = 100  # 神经元数量
        self.input_sz = 32     # 图像尺寸 (例如 28x28)
        self.hidden_size = 256 # LSTM 隐藏层大小
        self.num_layers = 2     # LSTM 层数
        self.enc_type = 'conv' # 编码器类型
        self.dec_type = 'conv' # 因为要改成 RNN 所以这里其实不重要
        self.dataset = 'MNIST' # 数据集类型
        self.n_ch = 32 # 通道数量
        self.prior_log_dist = 'uniform' # 可以改
        self.prior_clamp = -2.0 # 可以改
        self.seed = 123 # 随机数种子
        # 训练参数
        self.lr = 1e-3
        self.beta = 1.0  # KL 散度系数
        self.batch_size = 256
        self.epochs = 100
        self.grad_clip = 500.0
        self.log_freq = 10
        self.warmup_epochs = 5 # 可选项，如果需要
        self.use_warmup = False
        self.use_amp = False # 是否使用混合精度
        self.activation_fn = 'swish' 
        # 保存路径
        self.log_dir = 'logs'
        self.use_bn: bool = False,
        self.init_scale: float = 0.05,
        self.init_dist: str = 'Normal'