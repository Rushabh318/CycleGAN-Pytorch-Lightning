from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger():
    def __init__(self, tag, logdir, **kwargs):
        self.writer = SummaryWriter(log_dir=logdir, **kwargs)
        self.tag = tag
        
    def __call__(self, epoch, loss):
        self.writer.add_scalars(self.tag, loss, epoch)

    def release(self):
        self.writer.flush()
        self.writer.close()