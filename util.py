from __future__ import print_function

import pickle
import math
import glob
from functools import cache
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

CLASS_LOOKUP = {
    "AMINOGLYCOSIDE": 0,
    "BETA-LACTAM": 1,
    "FOLATE-SYNTHESIS-INHABITOR": 2,
    "GLYCOPEPTIDE": 3,
    "MACROLIDE": 4,
    "MULTIDRUG": 5,
    "PHENICOL": 6,
    "QUINOLONE": 7,
    "TETRACYCLINE": 8,
    "TRIMETHOPRIM": 9,
}



class CoalaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, run_type="train", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.run_type = run_type
        if self.run_type == "train":
            path_data = []
            root = "/home/tcoard/w/contrastive"
            with open(f"{root}/pairs_train.pkl", "rb") as f:
                train_pairs = pickle.load(f)

            for seqs in train_pairs:
                seqs = iter(seqs)
                for pair1 in seqs:
                    pair2 = next(seqs)
                    sep_path = pair1.split("|")
                    resistance = CLASS_LOOKUP[sep_path[2]]
                    path1 = f"{root}/{pair1}"
                    path2 = f"{root}/{pair2}"
                    path_data.append(((path1, path2), resistance))
            self.path_data = path_data

        elif self.run_type == "test":
            path_data = []
            root = "/home/tcoard/w/contrastive"
            with open(f"{root}/pairs_test.pkl", "rb") as f:
                test_pairs = pickle.load(f)

            for seqs in test_pairs:
                for seq in seqs:
                    if seq.endswith("var0.pt"):
                        sep_path = seq.split("|")
                        resistance = CLASS_LOOKUP[sep_path[2]]
                        path1 = f"{root}/{seq}"
                        path_data.append((path1, resistance))
            self.path_data = path_data



            # path_data = []
            # embedding_paths = glob.glob(f"{self.root_dir}/*var0.pt")
            # for path1 in embedding_paths:
            #     sep_path = path1.split("|")
            #     resistance = CLASS_LOOKUP[sep_path[2]]
            #     path_data.append((path1, resistance))
            # self.path_data = path_data



    def __len__(self):
        return len(self.path_data)


    # @property
    # def path_data(self):
    #     path_data = []
    #     embedding_paths = glob.glob(f"{self.root_dir}/*var1.pt")
    #     for path1 in embedding_paths:
    #         sep_path = path1.split("|")
    #         path2 = f"{'|'.join(sep_path[:-1])}|var2.pt"
    #         resistance = CLASS_LOOKUP[sep_path[2]]
    #         path_data.append(((path1, path2), resistance))
    #     return path_data

    #@cache
    def __getitem__(self, idx):
        if self.run_type == "train":
            (path1, path2), resistance = self.path_data[idx]
            return ((torch.load(path1).numpy(), torch.load(path2).numpy()), resistance)
        elif self.run_type == "test":
            path1, resistance = self.path_data[idx]
            return (torch.load(path1), resistance)



class BSWorkAround:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, path):
        # todo increment the pairs so that we can have more? The original doesn't do this
        # but they might still be able to make more???
        pair1 = torch.load(f"{path}|var1.pt")
        pair2 = torch.load(f"{path}|var2.pt")
        return [pair1, pair2]

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # print(x)
        # print(type(x))
        # print(vars(x))
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
