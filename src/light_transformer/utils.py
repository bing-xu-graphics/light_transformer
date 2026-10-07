import torch
import sys
import os
import numpy as np


def setup_device(dist):
    if dist:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda:0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device, local_rank


def index(pc, indx):
    device = pc.device
    B = pc.shape[0]
    view_shape = list(indx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    )
    new_pc = pc[batch_indices, indx, :]
    return new_pc


def dist_mat(pc1, pc2):
    B, N, _ = pc1.shape
    _, M, _ = pc2.shape
    dist = -2 * torch.matmul(pc1, pc2.permute(0, 2, 1))
    dist += torch.sum(pc1 ** 2, -1).view(B, N, 1)
    dist += torch.sum(pc2 ** 2, -1).view(B, 1, M)
    return dist


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, batch_metrics):
        if self.metrics == {}:
            for key, value in batch_metrics.items():
                self.metrics[key] = [value]
        else:
            for key, value in batch_metrics.items():
                self.metrics[key].append(value)

    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        avg_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])


def pbar(p=0, msg="", bar_len=20):
    sys.stdout.write("\033[K")
    sys.stdout.write("\x1b[2K" + "\r")
    block = int(round(bar_len * p))
    text = "Progress: [{}] {}% {}".format(
        "\x1b[32m" + "=" * (block - 1) + ">" + "\033[0m" + "-" * (bar_len - block),
        round(p * 100, 2),
        msg,
    )
    print(text, end="\r")
    if p == 1:
        print()


def print_args(args):
    print("\n---- experiment configuration ----")
    args_ = vars(args)
    for arg, value in args_.items():
        print(f" * {arg} => {value}")
    print("----------------------------------")


def tonemap(hdr_image):
    return torch.log(1.0 + hdr_image)
    # max_value = max(hdr_image.max(), 1e-5)
    # return torch.log1p(hdr_image) / torch.log1p(max_value)


class CosineScheduler:
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]
