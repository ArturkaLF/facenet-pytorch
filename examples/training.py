import logging
import torch
import numpy as np
import time
from apex import amp
from matplotlib import pyplot as plt


class Logger(object):

    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i):

        if i % (self.length // 5) == 0:
            track_str = '{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
            loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
            metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())

            logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')

            logging.info(track_str + loss_str + metric_str + '   ')


class BatchTimer(object):
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.
    
    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


def pass_epoch(
        model, loss_fn, loader, optimizer=None, scheduler=None,
        batch_metrics={'time': BatchTimer()}, show_running=True,
        device='cpu', writer=None, opt=0, classes=[], epoch=0
):
    """Train or evaluate over a data epoch.
    
    Arguments:
        model {torch.nn.Module} -- Pytorch model.
        loss_fn {callable} -- A function to compute (scalar) loss.
        loader {torch.utils.data.DataLoader} -- A pytorch data loader.
    
    Keyword Arguments:
        optimizer {torch.optim.Optimizer} -- A pytorch optimizer.
        scheduler {torch.optim.lr_scheduler._LRScheduler} -- LR scheduler (default: {None})
        batch_metrics {dict} -- Dictionary of metric functions to call on each batch. The default
            is a simple timer. A progressive average of these metrics, along with the average
            loss, is printed every batch. (default: {{'time': iter_timer()}})
        show_running {bool} -- Whether or not to print losses and metrics for the current batch
            or rolling averages. (default: {False})
        device {str or torch.device} -- Device for pytorch to use. (default: {'cpu'})
        writer {torch.utils.tensorboard.SummaryWriter} -- Tensorboard SummaryWriter. (default: {None})
    
    Returns:
        tuple(torch.Tensor, dict) -- A tuple of the average loss and a dictionary of average
            metric values across the epoch.
    """

    mode = 'Train' if model.training else 'Valid'
    logger = Logger(mode, length=len(loader), calculate_mean=show_running)
    loss = 0
    metrics = {}

    for i_batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss_batch = loss_fn(y_pred, y)

        if model.training:

            if opt:
                with amp.scale_loss(loss_batch, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_batch.backward()

            optimizer.step()
            optimizer.zero_grad()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(y_pred, y).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

        if writer is not None and model.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss_batch.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        # Check error in classification
        EPOCHS_FOR_CHECK = 8
        ACC_FOR_CHECK = 0.96
        if 1.0 > accuracy(y_pred, y) > ACC_FOR_CHECK and not model.training and epoch > EPOCHS_FOR_CHECK:
            for i in range(int(y.size()[0])):
                if torch.max(y_pred, 1)[1][i] != y[i]:
                    imshow(img=x[i].cpu(), y_pred=classes[int(torch.max(y_pred, 1)[1][i])], y=classes[int(y[i])])
                    print("Add a new image with error")

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, metrics, i_batch)
        else:
            logger(loss_batch, metrics_batch, i_batch)

    if model.training and scheduler is not None:
        scheduler.step()

    loss = loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

    if writer is not None and not model.training:
        writer.add_scalars('loss', {mode: loss.detach()}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return loss, metrics


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


def imshow(img, y_pred, y):
    img = img / 2 + 0.5
    npigm = img.numpy()
    plt.imshow(np.transpose(npigm, (1, 2, 0)))
    plt.title(f"Y:{y} -- Y_pred:{y_pred}")
    plt.savefig(f"../Logs/imgs_test_8.1/Y:{y} -- Y_pred:{y_pred}.png")
