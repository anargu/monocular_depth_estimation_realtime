import math
from torchvision import transforms
from datasets.utils import create_transforms
from datasets.nyuv2 import NYUv2
import torch
import time
import numpy as np
from model import RealTimeDepth
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/depth_estimation_experiment_1')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


m = 4
M = 80

C = lambda y, y_hat: (1/5) * np.max(np.abs(G(y_hat) - G(y)))

G = lambda d: ((np.log(d) - np.log(m)) * M)/(np.log(M)-np.log(m))

F = lambda x, c : np.abs(x) if np.abs(x) <= c else (x**2 + c**2)/(2*c) 

y_actual_size = (400,400)
transform_y_pred = transforms.Resize(y_actual_size)
def loss_d (y_pred, y_target):
    y_pred = transform_y_pred(y_pred)
    assert y_pred.shape == y_target.shape
    e = y_pred.shape[0]
    w = y_pred.shape[2]
    h = y_pred.shape[3]

    batch_loss = []
    for q in range(e):
        sumF = 0
        for i in range(w):
            for j in range(h):
                y = (y_target[q][0][i][j]).item()
                y_hat = (y_pred[q][0][i][j]).item()

                c = C(y, y_hat)
                gy_hat = G(y_hat)
                gy = G(y)
                sumF += F(gy_hat - gy, c)
        batch_loss.append((sumF)/(w*h)) 
    loss = np.mean(batch_loss)
    return torch.tensor(loss)
    
def inverse_huber_loss(output,target):
    output = transform_y_pred(output)
    absdiff = torch.abs(output-target)
    C = 0.2*torch.max(absdiff).item()
    return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))

crossentropyloss = torch.nn.CrossEntropyLoss()
def loss_s (y_pred, y_actual):
    # y_pred, _ = torch.max(y_pred, dim=1, keepdim=True)
    y_pred = transform_y_pred(y_pred)
    y_actual = torch.argmax(y_actual, dim=1)
    # assert y_pred.shape == y_actual.shape
    # e = y_pred.shape[0]
    # c = y_pred.shape[1]
    # w = y_pred.shape[2]
    # h = y_pred.shape[3]

    # computation of loss on segmentation data along the batch size
    # log_yactual = torch.log10(y_actual)
    # sum_y = torch.sum(y_pred * log_yactual, dim=(2,3))
    # loss_seg = (-1) * (sum_y) / ( w * h )

    # batch_loss = []
    #     # aggregate = 0
    # for i in range(w):
    #     for j in range(h):
    #         for q in range(e):
    #             y = (y_actual[:,0,i,j]).item()
    #             y_hat = (y_pred[:,0,i,j]).item()


    #             aggregate += y_hat * np.log(y)
    #     batch_loss.append(-1 * (aggregate)/(w*h)) 
    # loss = np.mean(batch_loss)
    total_loss = crossentropyloss(y_pred, y_actual)
    # mean_loss_seg = total_loss.mean()
    return torch.tensor(total_loss)

a = 0.25
b = 0.75
def loss_fn(y_pred, y_seg_actual, y_depth_actual):
    # y_pred[0] depth
    # y_pred[1] seg
    lossd = inverse_huber_loss(y_pred[0], y_depth_actual)
    losss = loss_s(y_pred[1], y_seg_actual)
    return (a * lossd) + (b * losss)


def compute_errors(output, target):
    valid_mask = ((target>0) + (output>0)) > 0

    output = 1e3 * output[valid_mask]
    target = 1e3 * target[valid_mask]
    abs_diff = (output - target).abs()

    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    mae = float(abs_diff.mean())
    # self.lg10 = float((log10(output) - log10(target)).abs().mean())
    absrel = float((abs_diff / target).mean())

    return mse, rmse, mae, absrel

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    running_loss = 0
    mse, rmse, mae, absrel = 0, 0, 0, 0

    for i, (images, target_seg, target_depth) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args['gpu'] is not None:
            images = images.cuda(args['gpu'], non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args['gpu'], non_blocking=True)

        # compute output
        output = model(images)
        output_depth = output[0].detach().cpu().numpy()
        np.save(f'predicted_depth_{i}.npy', output_depth)
        output_seg = output[1].detach().cpu().numpy()
        np.save(f'predicted_seg_{i}.npy', output_seg)
        
        loss = criterion(output, target_seg, target_depth)

        # measure accuracy and record loss

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # ...log the running loss
        running_loss += loss.item()
        mse, rmse, mae, absrel += compute_errors(output[0], target_depth)
        if i % args['print_freq'] == 0:
            writer.add_scalar('mse',
                        mse / args['print_freq'],
                        epoch * len(train_loader) + i)
            writer.add_scalar('rmse',
                        rmse / args['print_freq'],
                        epoch * len(train_loader) + i)
            writer.add_scalar('absrel',
                        absrel / args['print_freq'],
                        epoch * len(train_loader) + i)

            writer.add_scalar('training loss',
                        running_loss / args['print_freq'],
                        epoch * len(train_loader) + i)

            running_loss = 0
            mse, rmse, absrel, mae = 0, 0, 0, 0
            progress.display(i)


if __name__ == '__main__':

    # root_dataset = '/Volumes/New Volume/ml/datasets/nyuv2/'
    data_path = '/Volumes/New Volume/ml/datasets/nyuv2/nyu_depth_v2_labeled.mat' 

    model = RealTimeDepth(True, 1)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    t = create_transforms()
    t2 = create_transforms()
    t3 = create_transforms()

    nyu_dataset = NYUv2(
        root=data_path,
        train=True,
        download=False,
        rgb_transform=t,
        depth_transform=t2,
        seg_transform=t3
    )

    train_dataloader = DataLoader(nyu_dataset, batch_size=8, shuffle=True)
    train(train_loader=train_dataloader, model=model, criterion=loss_fn,  optimizer=optimizer, epoch=20, args={'gpu': None, 'print_freq': 10})
