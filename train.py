import torch
import random
import numpy as np
import os
import json
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split

from light_transformer.config import DATA_CONFIG
from light_transformer.dataset import SpatialDataset_chunks
from light_transformer.utils import setup_device, AvgMeter, tonemap, pbar, CosineScheduler
from light_transformer.model import Model

def RelL2(outputs, targets):
    sg_outputs = outputs.detach().clone()
    loss = torch.norm(outputs - targets, p=2) / (torch.norm(sg_outputs, p=2) + 0.01)
    return loss


def read_scene_ids(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"scene list not found: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        return np.array(sorted(line.strip() for line in stream if line.strip()))

class Trainer:
    def __init__(self, args):
        self.args = args
        self.out_dir = args.out_dir

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.device, local_rank = setup_device(args.dist)
        self.main_thread = True if local_rank == 0 else False
        if self.main_thread:
            print(f"\nsetting up device, distributed = {args.dist}")
        print(f" | {self.device}")

        self.prepare_data()
        model = Model(
            DATA_CONFIG=DATA_CONFIG, 
            depth=args.depth,
            dim=args.dim,
            k=args.k,
        )
        if args.dist:
            torch.set_num_threads(1)
            self.model = DistributedDataParallel(
                model.to(self.device),
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            self.model = model.to(self.device)
        if self.main_thread:
            print(f"# of model parameters: {sum(p.numel() for p in self.model.parameters())/1e6}M")

        self.criterion = RelL2
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        # self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, args.total_steps)
        # self.lr_sched = torch.optim.lr_scheduler.StepLR(self.optim, step_size=50_000, gamma=0.5)
        self.lr_sched = CosineScheduler(
            base_value=args.lr,
            final_value=1e-6,
            total_iters=args.total_steps,
            warmup_iters=50_000,
            start_warmup_value=0,
        )
        
        if os.path.exists(os.path.join(args.out_dir, "last.ckpt")):
            if not args.resume:
                raise ValueError(
                    f"directory {args.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(os.path.join(args.out_dir, "last.ckpt"), map_location=self.device)
            model_dict = ckpt["model"]
            if "module" in list(model_dict.keys())[0] and args.dist == False:
                model_dict = {
                    key.replace("module.", ""): value for key, value in model_dict.items()
                }
            self.model.load_state_dict(model_dict)
            self.optim.load_state_dict(ckpt["optim"])
            # self.lr_sched.load_state_dict(ckpt["lr_sched"])
            self.start_step = ckpt["epoch"] + 1
            if self.main_thread:
                print(
                    f"loaded checkpoint, resuming training expt from {self.start_step} to {args.total_steps} steps."
                )
        else:
            if args.resume:
                raise ValueError(
                    f"resume training args are true but no checkpoint found in {args.out_dir}"
                )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location=self.device)
                model_dict = ckpt["model"]
                if "module" in list(model_dict.keys())[0] and args.dist == False:
                    model_dict = {
                        key.replace("module.", ""): value for key, value in model_dict.items()
                    }
                new_model_dict = {}
                for key, value in model_dict.items():
                    if "embedding" in key:
                        continue
                    if "out_fc" in key:
                        continue
                    new_model_dict[key] = value
                self.model.load_state_dict(new_model_dict, strict=False)
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
            os.makedirs(os.path.join(args.out_dir, "imgs"), exist_ok=True)
            self.start_step = 0
            if self.main_thread:
                print(f"starting fresh training expt for {args.total_steps} steps.")
                if args.pretrain_ckpt:
                    print(f"using pretraining checkpoint from {args.pretrain_ckpt}")
        
        self.metric_meter = AvgMeter()
        if self.main_thread:
            self.log_f = open(os.path.join(args.out_dir, "logs.txt"), "w")
            print(f"start file logging @ {os.path.join(args.out_dir, 'logs.txt')}")
            self.writer = None
            if args.tb:
                self.writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'tb'))
    

    def prepare_data(self):
        scene_ids = read_scene_ids(self.args.scenes_list)
        if self.main_thread:
            print(f"Total data samples found: {len(scene_ids)}")

        if self.args.test_size == 0:
            train_scenes = scene_ids
            val_scenes = scene_ids
        else:
            train_scenes, val_scenes = train_test_split(scene_ids, test_size=self.args.test_size, random_state=self.args.seed)
            
        train_paths_scene_points = [os.path.join(self.args.scenes_dir, f"scene_{scene_id}_frame_0_data_scene_data.npy") for scene_id in train_scenes]
        train_paths_query_points = [os.path.join(self.args.query_dir, f"scene_{scene_id}_frame_0_sampled_data_query_point_feat.npy") for scene_id in train_scenes]
        train_paths_query_field = [os.path.join(self.args.query_field_dir, f"scene_{scene_id}_frame_0_sampled_query_point_irradiance.npy") for scene_id in train_scenes]
        
        val_paths_scene_points = [os.path.join(self.args.scenes_dir, f"scene_{scene_id}_frame_0_data_scene_data.npy") for scene_id in val_scenes]
        val_paths_query_points = [os.path.join(self.args.query_dir, f"scene_{scene_id}_frame_0_sampled_data_query_point_feat.npy") for scene_id in val_scenes]
        val_paths_query_field = [os.path.join(self.args.query_field_dir, f"scene_{scene_id}_frame_0_sampled_query_point_irradiance.npy") for scene_id in val_scenes]

        train_dataset = SpatialDataset_chunks(train_paths_scene_points, train_paths_query_points, train_paths_query_field, self.args.scene_points, self.args.query_points, randomize=True)
        val_dataset = SpatialDataset_chunks(val_paths_scene_points, val_paths_query_points, val_paths_query_field, self.args.scene_points, self.args.query_points, randomize=False)
        if self.main_thread:
            print(f"train data: {len(train_dataset)}, val data: {len(val_dataset)}")
        
        if self.args.dist:
            train_sampler = DistributedSampler(train_dataset)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                num_workers=self.args.n_workers,
            )
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.n_workers,
            )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.n_workers,
        )
    
    def train_step(self, data):
        scene_data, query_data_features, query_data_values = data
        scene_data = scene_data.to(self.device)
        query_data_features = query_data_features.to(self.device)
        query_data_values = query_data_values.to(self.device)

        out, extra_loss = self.model(scene_data, query_data_features)
        # out, mat_diff_loss = self.model(scene_data, query_data_features)

        query_data_values = tonemap(query_data_values)
        out = tonemap(out)

        self.optim.zero_grad()
        loss = self.criterion(out, query_data_values)
        for l in extra_loss.values():
            loss = loss + l
        loss.backward()
        self.optim.step()

        metrics = {
            "train_loss": loss.item(),
            # "mat_diff_loss": mat_diff_loss.item()
        }
        for k, l in extra_loss.items():
            metrics[k] = l.item()
        
        self.metric_meter.add(metrics)

        if self.main_thread and self.train_steps % self.args.log_every == 0:
            if self.writer:
                self.writer.add_scalar('Loss/Train', loss.item(), self.train_steps)
            pbar(self.train_steps / self.args.total_steps, msg=self.metric_meter.msg())
    
    @torch.no_grad()
    def eval(self):
        self.metric_meter.reset()
        self.model.eval()
        for indx, (scene_data, query_data_features, query_data_values) in enumerate(self.val_loader):
            scene_data = scene_data.to(self.device)
            query_data_features = query_data_features.to(self.device)
            query_data_values = query_data_values.to(self.device)

            out, _ = self.model(scene_data, query_data_features)

            query_data_values = tonemap(query_data_values)
            out = tonemap(out)

            loss = self.criterion(out, query_data_values)
            metrics = {
                "val_loss": loss.item(),
            }
            self.metric_meter.add(metrics)
            pbar(indx / len(self.val_loader), msg=self.metric_meter.msg())
        if self.writer:
            self.writer.add_scalar('Loss/Val', self.metric_meter.get()["val_loss"], self.train_steps)
        pbar(1, msg=self.metric_meter.msg())
    
    def train(self):
        best_train, best_val = float("inf"), float("inf")
        train_data_iter = iter(self.train_loader)

        for self.train_steps in range(self.start_step, self.args.total_steps):
            try:
                data = next(train_data_iter)
            except StopIteration:
                self.metric_meter.reset()
                train_data_iter = iter(self.train_loader)
                data = next(train_data_iter)
            self.train_step(data)

            if self.main_thread and (self.train_steps) % self.args.eval_every == 0:
                print("\n")
                print("-----")
                train_metrics = self.metric_meter.get()
                if train_metrics["train_loss"] < best_train:
                    print(
                        "\x1b[34m"
                        + f"train loss improved from {round(best_train, 5)} to {round(train_metrics['train_loss'], 5)}"
                        + "\033[0m"
                    )
                    best_train = train_metrics["train_loss"]
                msg = f"train step: {self.train_steps}, last train: {round(train_metrics['train_loss'], 5)}, best train: {round(best_train, 5)}"

                # val_metrics = {}
                # self.eval()
                # val_metrics = self.metric_meter.get()
                # if val_metrics["val_loss"] < best_val:
                #     print(
                #         "\x1b[33m"
                #         + f"val loss improved from {round(best_val, 5)} to {round(val_metrics['val_loss'], 5)}"
                #         + "\033[0m"
                #     )
                #     best_val = val_metrics["val_loss"]
                #     torch.save(
                #         self.model.state_dict(),
                #         os.path.join(self.args.out_dir, f"best.ckpt"),
                #     )
                # msg += f", last val: {round(val_metrics['val_loss'], 5)}, best val: {round(best_val, 5)}"
                # self.infer()

                print(msg)
                self.log_f.write(msg + f", lr: {round(self.optim.param_groups[0]['lr'], 5)}\n")
                self.log_f.flush()

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        # "lr_sched": self.lr_sched.state_dict(),
                        "epoch": self.train_steps,
                    },
                    os.path.join(self.args.out_dir, "last.ckpt"),
                )
                self.model.train()
            
            # self.lr_sched.step()
            new_lr = self.lr_sched[self.train_steps]
            for param_group in self.optim.param_groups:
                param_group['lr'] = new_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"outputs/{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
    )
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument(
        "--dist", action="store_true",
    )
    parser.add_argument("--scenes_list", type=str, required=True,)
    parser.add_argument("--scenes_dir", type=str, required=True,)
    parser.add_argument("--query_dir", type=str, required=True,)
    parser.add_argument("--query_field_dir", type=str, required=True,)
    parser.add_argument("--test_size", type=float, default=0.1,)
    parser.add_argument(
        "--scene_points", type=int, default=20_000,
    )
    parser.add_argument(
        "--query_points", type=int, default=8_192,
    )
    parser.add_argument("--batch_size", type=int, default=3,)
    parser.add_argument(
        "--n_workers", type=int, default=4,
    )
    parser.add_argument(
        "--depth", type=int, default=6,
    )
    parser.add_argument(
        "--dim", type=int, default=128,
    )
    parser.add_argument(
        "--k", type=int, default=32,
    )
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--total_steps", type=int, default=2_000_000,)
    parser.add_argument("--eval_every", type=int, default=10_000,)
    parser.add_argument("--log_every", type=int, default=10,)
    parser.add_argument("--pretrain_ckpt", type=str, default=None,)
    parser.add_argument(
        "--resume", action="store_true",
    )
    parser.add_argument(
        "--tb", action="store_true",
    )
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

    if args.dist:
        torch.distributed.destroy_process_group()
