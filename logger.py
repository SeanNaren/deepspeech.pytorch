import os

import torch


def to_np(x):
    return x.cpu().numpy()


class VisdomLogger(object):
    def __init__(self, id, num_epochs):
        from visdom import Visdom
        self.viz = Visdom()
        self.opts = dict(title=id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        self.viz_window = None
        self.epochs = torch.arange(1, num_epochs + 1)
        self.visdom_plotter = True

    def update(self, epoch, values):
        x_axis = self.epochs[0:epoch + 1]
        y_axis = torch.stack((values["loss_results"][:epoch + 1],
                              values["wer_results"][:epoch + 1],
                              values["cer_results"][:epoch + 1]),
                             dim=1)
        self.viz_window = self.viz.line(
            X=x_axis,
            Y=y_axis,
            opts=self.opts,
            win=self.viz_window,
            update='replace' if self.viz_window else None
        )

    def load_previous_values(self, start_epoch, package):
        self.update(start_epoch - 1, package)  # Add all values except the iteration we're starting from


class TensorBoardLogger(object):
    def __init__(self, id, log_dir, log_params):
        os.makedirs(log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)
        self.log_params = log_params

    def update(self, epoch, values, parameters=None):
        loss, wer, cer = values["loss_results"][epoch + 1], values["wer_results"][epoch + 1], \
                         values["cer_results"][epoch + 1]
        values = {
            'Avg Train Loss': loss,
            'Avg WER': wer,
            'Avg CER': cer
        }
        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)
        if self.log_params:
            for tag, value in parameters():
                tag = tag.replace('.', '/')
                self.tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                self.tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)

    def load_previous_values(self, start_epoch, values):
        loss_results = values["loss_results"][:start_epoch]
        wer_results = values["wer_results"][:start_epoch]
        cer_results = values["cer_results"][:start_epoch]

        for i in range(start_epoch):
            values = {
                'Avg Train Loss': loss_results[i],
                'Avg WER': wer_results[i],
                'Avg CER': cer_results[i]
            }
            self.tensorboard_writer.add_scalars(self.id, values, i + 1)
