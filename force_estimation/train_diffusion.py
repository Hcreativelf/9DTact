"""
Training script for Diffusion-based 6D Force Estimation.

This script replaces the original CNN-based train.py with a diffusion model
that treats force prediction as a conditional denoising process.

Key differences from the original train.py:
1. Model: DiffusionForceEstimator (CNN encoder + MLP denoiser) instead of plain CNN
2. Loss: MSE on predicted noise (standard DDPM loss) instead of L1 on force
3. Training: sample random timestep t, add noise to force, predict noise
4. Inference: run full reverse diffusion with optional ensemble averaging
5. Fixed: original code had `predict_forces[predict_forces < 0] = 0` during training
   which breaks gradient flow through in-place ops; we use clamp properly only at eval.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import yaml
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim, nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

np.set_printoptions(suppress=True)
import argparse
import cv2

from data import DTactDataset
from diffusion_model import DiffusionForceEstimator, GaussianDiffusion

# ---- wandb toggle ----
use_wandb = True

# ---- Wrench normalization range ----
wrench_range = [[-3, 3], [-3, 3], [-12, 0], [-0.2, 0.2], [-0.2, 0.2], [-0.05, 0.05]]
min_wrench = np.array([wr[0] for wr in wrench_range], dtype=np.float32)
max_wrench = np.array([wr[1] for wr in wrench_range], dtype=np.float32)
wrench_span = max_wrench - min_wrench


def get_time_dif(start_time):
    return time.time() - start_time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class DiffusionForceEstimationTrainer:
    def __init__(self, cfg):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", default="Densenet", type=str,
                            help="CNN backbone: Resnet or Densenet")
        parser.add_argument("--model_layer", default=169, type=int,
                            help="Backbone layer depth")
        parser.add_argument("--optimizer", default="ADAM", type=str)
        parser.add_argument("--lrs", default=True, type=str2bool,
                            help="Use learning rate scheduler")
        parser.add_argument("--image_type", default="RGB", type=str)
        parser.add_argument("--cuda_index", default=0, type=int)
        parser.add_argument("--train_mode", default=True, type=str2bool)
        parser.add_argument("--test_object", default=False, type=str2bool)
        parser.add_argument("--mixed_image", default=True, type=str2bool)
        parser.add_argument("--pretrained", default=True, type=str2bool)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--num_epoch", default=200, type=int)
        parser.add_argument("--learning_rate", default=5e-4, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--end_factor", default=0.4, type=float)
        parser.add_argument("--total_iters", default=20, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        # Diffusion-specific arguments
        parser.add_argument("--num_timesteps", default=1000, type=int,
                            help="Number of diffusion timesteps")
        parser.add_argument("--beta_start", default=1e-4, type=float)
        parser.add_argument("--beta_end", default=0.02, type=float)
        parser.add_argument("--feature_dim", default=256, type=int,
                            help="Dimension of CNN condition features")
        parser.add_argument("--hidden_dim", default=512, type=int,
                            help="Hidden dimension of denoising MLP")
        parser.add_argument("--num_mlp_layers", default=6, type=int,
                            help="Number of residual blocks in denoiser")
        parser.add_argument("--num_ensemble", default=10, type=int,
                            help="Ensemble size during inference")
        parser.add_argument("--use_ddim", default=True, type=str2bool,
                            help="Use DDIM for faster inference")
        parser.add_argument("--ddim_steps", default=50, type=int,
                            help="Number of DDIM steps (if use_ddim=True)")
        parser.add_argument("--resize_img", default=False, type=str2bool)
        parser.add_argument("--eval_batch", default=32, type=int)
        args = parser.parse_args()

        # ---- Resolve parameters ----
        model_name = args.model_name
        model_layer = args.model_layer
        model_type = f"Diffusion-{model_name}-{model_layer}"
        image_type = args.image_type
        cuda_index = args.cuda_index
        train_mode = args.train_mode
        test_object = args.test_object
        mixed_image = args.mixed_image
        pretrained = args.pretrained and train_mode
        batch_size = args.batch_size
        eval_batch = args.eval_batch if not train_mode else batch_size
        self.lrs = args.lrs
        self.num_ensemble = args.num_ensemble
        self.use_ddim = args.use_ddim
        self.ddim_steps = args.ddim_steps

        # ---- Device ----
        self.device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
        print(f"[Device] {self.device}")

        # ---- Build model ----
        self.model = DiffusionForceEstimator(
            backbone=model_name,
            layer=model_layer,
            pretrained=pretrained,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_mlp_layers=args.num_mlp_layers,
        ).to(self.device)

        # ---- Build diffusion scheduler ----
        self.diffusion = GaussianDiffusion(
            num_timesteps=args.num_timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=self.device,
        )

        # ---- Loss: MSE on predicted noise ----
        self.loss_fn = nn.MSELoss(reduction='mean')

        # ---- Wrench range on device ----
        self.min_wrench = torch.tensor(min_wrench, dtype=torch.float32).to(self.device)
        self.wrench_span = torch.tensor(wrench_span, dtype=torch.float32).to(self.device)

        # ---- Test dataset ----
        test_dataset = DTactDataset(
            mode='test', root_path=cfg['data_dir'], image_type=image_type,
            test_object=test_object, mixed_image=mixed_image
        )
        self.test_img_num = len(test_dataset)
        self.test_dataLoader = DataLoader(
            test_dataset, batch_size=eval_batch, shuffle=False, num_workers=args.num_workers
        )

        if train_mode:
            self.num_epoch = args.num_epoch
            learning_rate = args.learning_rate
            weight_decay = args.weight_decay

            # ---- Create directories ----
            time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
            self.save_dir = cfg['save_dir'] + "/" + model_type + time_suffix
            self.log_dir = self.save_dir + '/log'
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)

            # ---- Save config ----
            self.write_basic_information(
                model_type, batch_size, self.num_epoch, learning_rate, weight_decay,
                args.num_timesteps, args.feature_dim, args.hidden_dim, args.num_mlp_layers
            )

            # ---- Train dataset ----
            train_dataset = DTactDataset(
                mode='train', root_path=cfg['data_dir'], image_type=image_type,
                test_object=test_object, mixed_image=mixed_image
            )
            self.train_img_num = len(train_dataset)
            self.train_dataLoader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
            )

            # ---- Optimizer ----
            if args.optimizer.upper() == 'SGD':
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
            else:
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=learning_rate,
                    betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
                )
            self.scheduler = lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0,
                end_factor=args.end_factor, total_iters=args.total_iters
            )

            # ---- wandb ----
            wandb_config = {
                "model_type": model_type,
                "image_type": image_type,
                "optimizer": args.optimizer,
                "test_object": test_object,
                "mixed_image": mixed_image,
                "pretrained": pretrained,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epoch": self.num_epoch,
                "weight_decay": weight_decay,
                "num_timesteps": args.num_timesteps,
                "feature_dim": args.feature_dim,
                "hidden_dim": args.hidden_dim,
                "num_mlp_layers": args.num_mlp_layers,
                "num_ensemble": self.num_ensemble,
            }
            run_name = f"{model_type}_lr-{learning_rate:.4f}_T{args.num_timesteps}"
            if use_wandb:
                import wandb
                self.run_log = wandb.init(
                    project="Diffusion_Force_Estimation", config=wandb_config, name=run_name
                )

            # ---- Start training ----
            self.train()
        else:
            # ---- Evaluation mode ----
            weights_idx = cfg['model_choice']
            self.save_dir = cfg['save_dir'] + '/' + model_type + str(cfg['weights'][weights_idx][0])
            weights_path = self.save_dir + '/epoch_' + str(cfg['weights'][weights_idx][1]) + '.pt'
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"[Eval] Loaded weights from {weights_path}")
            self.test_record(cfg['weights'][weights_idx][1])

    def write_basic_information(self, model_type, batch_size, num_epoch, learning_rate,
                                weight_decay, num_timesteps, feature_dim, hidden_dim, num_mlp_layers):
        path = self.save_dir + "/basic.txt"
        with open(path, 'w') as f:
            f.write(f"model: {model_type}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"epoch: {num_epoch}\n")
            f.write(f"learning_rate: {learning_rate}\n")
            f.write(f"weight_decay: {weight_decay}\n")
            f.write(f"num_timesteps: {num_timesteps}\n")
            f.write(f"feature_dim: {feature_dim}\n")
            f.write(f"hidden_dim: {hidden_dim}\n")
            f.write(f"num_mlp_layers: {num_mlp_layers}\n")

    def train(self):
        start_time = time.time()
        test_best_loss = float('inf')
        iter_num = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        train_txt_path = self.save_dir + "/train.txt"
        train_txt_file = open(train_txt_path, 'a')
        best_epoch = 0

        print(f"[Train] Starting training for {self.num_epoch} epochs")
        print(f"[Train] Train samples: {self.train_img_num}, Test samples: {self.test_img_num}")

        for epoch in range(self.num_epoch):
            epoch_loss = 0.0
            num_batches = 0
            self.model.train()

            for batch_idx, dataset in enumerate(self.train_dataLoader):
                images, label_forces = dataset
                images = images.float().to(self.device)
                label_forces = label_forces.float().to(self.device)

                # ---- Diffusion training step ----
                # 1) Sample random timesteps
                batch_size_curr = images.size(0)
                t = torch.randint(
                    0, self.diffusion.num_timesteps,
                    (batch_size_curr,), device=self.device, dtype=torch.long
                )

                # 2) Add noise to the force (forward diffusion)
                x_noisy, noise = self.diffusion.q_sample(label_forces, t)

                # 3) Predict the noise
                noise_pred = self.model(x_noisy, t, images)

                # 4) Compute loss
                loss = self.loss_fn(noise_pred, noise)

                # 5) Backprop
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                iter_num += 1

            # ---- Learning rate schedule ----
            lr = self.optimizer.param_groups[0]["lr"]
            if self.lrs:
                self.scheduler.step()

            avg_train_loss = epoch_loss / num_batches

            # ---- Evaluate ----
            time_dif = get_time_dif(start_time)
            test_loss, force_test_error, force_test_error_span = self.test()

            if test_loss < test_best_loss:
                test_best_loss = test_loss
                torch.save(self.model.state_dict(), self.save_dir + f"/epoch_{epoch + 1}.pt")
                improve = '******'
                best_epoch = epoch + 1
            else:
                improve = ''

            msg = (
                f'[Epoch: {epoch + 1}/{self.num_epoch}, iter: {iter_num}], '
                f'Lr: {lr:.6f}, Train Noise Loss: {avg_train_loss:.6f}, '
                f'Test Force MAE: {test_loss:.4f}, '
                f'Test Error: {np.round(force_test_error.cpu().numpy(), 4)}, '
                f'Test Error(span): {np.round(force_test_error_span.cpu().numpy(), 3)}, '
                f'Time: {time_dif / 60:.1f}min {improve}'
            )
            print(msg)

            writer.add_scalar("loss/train_noise", avg_train_loss, iter_num)
            writer.add_scalar("loss/test_force_mae", test_loss, iter_num)
            train_txt_file.write(msg + "\n")

            if use_wandb:
                log_dict = {
                    "train_noise_loss": avg_train_loss,
                    "test_force_mae": test_loss,
                    "lr": lr,
                }
                for i, name in enumerate(['fx', 'fy', 'fz', 'tx', 'ty', 'tz']):
                    log_dict[f"test_error_{name}"] = force_test_error[i].item()
                self.run_log.log(log_dict)

            self.model.train()

        writer.close()
        train_txt_file.close()
        if use_wandb:
            self.run_log.finish()
        self.test_record(best_epoch)
        print(f"[Train] Done. Best epoch: {best_epoch}")

    @torch.no_grad()
    def test(self):
        """
        Evaluate by running reverse diffusion on test set.
        Returns:
            avg_mae:             average MAE over normalized forces
            force_test_error:    per-component MAE (normalized)
            force_test_error_span: per-component MAE in physical units
        """
        self.model.eval()
        force_test_error = torch.zeros(6, dtype=torch.float32, device=self.device)
        total_samples = 0

        for batch_idx, dataset in enumerate(self.test_dataLoader):
            images, label_forces, image_names = dataset
            images = images.float().to(self.device)
            label_forces = label_forces.float().to(self.device)

            # Predict force via reverse diffusion (use fewer ensemble for speed during training)
            predict_forces = self.model.predict_force(
                images, self.diffusion, num_ensemble=3,
                use_ddim=self.use_ddim, ddim_steps=self.ddim_steps
            )

            # Clamp predictions to [0, 1] (normalized range)
            predict_forces = predict_forces.clamp(0, 1)

            force_test_error += torch.sum(torch.abs(predict_forces - label_forces), dim=0)
            total_samples += images.size(0)

        force_test_error = force_test_error / total_samples
        force_test_error_span = force_test_error * self.wrench_span
        avg_mae = force_test_error.mean().item()

        return avg_mae, force_test_error, force_test_error_span

    @torch.no_grad()
    def test_record(self, epoch):
        """Detailed evaluation and recording, same functionality as original."""
        self.model.eval()
        test_txt_path = self.save_dir + f"/test_{epoch}.txt"
        if os.path.exists(test_txt_path):
            os.remove(test_txt_path)
        test_txt_file = open(test_txt_path, 'a')
        test_image_dir = self.save_dir + f"/image_{epoch}"
        os.makedirs(test_image_dir, exist_ok=True)

        image_name_list = []
        mean_error_list = []
        truth_list = []
        predict_list = []
        error_list = []

        start_time = time.time()
        for batch_idx, dataset in enumerate(self.test_dataLoader):
            images, label_forces, image_paths = dataset
            images = images.float().to(self.device)
            label_forces = label_forces.float().to(self.device)

            # Full ensemble inference
            predict_forces = self.model.predict_force(
                images, self.diffusion, num_ensemble=self.num_ensemble,
                use_ddim=self.use_ddim, ddim_steps=self.ddim_steps
            )
            predict_forces = predict_forces.clamp(0, 1)

            # Convert to physical units
            force_truth = label_forces * self.wrench_span + self.min_wrench
            force_predict = predict_forces * self.wrench_span + self.min_wrench
            force_error = torch.abs(force_truth - force_predict)

            for i in range(images.size(0)):
                msg = (
                    f"{len(image_name_list) + 1}: "
                    f"{image_paths[i].split('image')[-1]} "
                    f"Mean: {force_error[i].mean():.4f} "
                    f"Tru: {np.round(force_truth[i].cpu().numpy(), 4)} "
                    f"Pre: {np.round(force_predict[i].cpu().numpy(), 4)} "
                    f"Err: {np.round(force_error[i].cpu().numpy(), 4)}"
                )
                test_txt_file.write(msg + "\n")
                image_name_list.append(image_paths[i])
                mean_error_list.append(force_error[i].mean().item())
                truth_list.append(force_truth[i].cpu().numpy())
                predict_list.append(force_predict[i].cpu().numpy())
                error_list.append(force_error[i].cpu().numpy())

        test_txt_file.write("--------------------------------------------\n")
        time_dif = get_time_dif(start_time)
        final_error = np.mean(error_list, axis=0)
        msg_time = f"Final Error: {final_error}, Time: {time_dif:.2f}s, FPS: {self.test_img_num / time_dif:.2f}\n"
        print(msg_time)
        test_txt_file.write(msg_time)
        test_txt_file.write("--------------------------------------------\n")

        # Sort by error
        sorted_index = np.argsort(np.array(mean_error_list))
        for j in range(len(error_list)):
            idx = sorted_index[j]
            msg_sort = (
                f"{j + 1}: {image_name_list[idx].split('image')[-1]} "
                f"Tru: {np.round(truth_list[idx], 4)} "
                f"Pre: {np.round(predict_list[idx], 4)} "
                f"Err: {np.round(error_list[idx], 4)}"
            )
            test_txt_file.write(msg_sort + "\n")
            saved_img = cv2.imread(image_name_list[idx])
            if saved_img is not None:
                cv2.imwrite(test_image_dir + f'/{j + 1}.png', saved_img)

        truth_arr = np.array(truth_list)
        predict_arr = np.array(predict_list)
        np.save(test_image_dir + '/truth_list.npy', truth_arr)
        np.save(test_image_dir + '/predict_list.npy', predict_arr)
        test_txt_file.close()
        print(f"[Test Record] Saved to {self.save_dir}")


if __name__ == '__main__':
    f = open("force_config.yaml", 'r+', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = DiffusionForceEstimationTrainer(config)
