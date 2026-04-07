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
import torchvision.models as models

np.set_printoptions(suppress=True)
import argparse
from data import DTactDataset
from model import *
import cv2

# use_wandb = True
use_wandb = True

wrench_range = [[-3, 3], [-3, 3], [-12, 0], [-0.2, 0.2], [-0.2, 0.2], [-0.05, 0.05]]
min_wrench = np.array([.0, .0, .0, .0, .0, .0])
max_wrench = np.array([.0, .0, .0, .0, .0, .0])
for i in range(len(wrench_range)):
    min_wrench[i] = wrench_range[i][0]
    max_wrench[i] = wrench_range[i][1]
wrench_span = max_wrench - min_wrench


class DiffusionForceModel(nn.Module):
    def __init__(self, model_name='Resnet', layer=50, pretrained=True, feature_dim=512):
        super(DiffusionForceModel, self).__init__()
        
        # 图像编码器 - 使用预训练的CNN作为条件编码器
        if model_name == 'Resnet':   # 注意：这里需要能获取到 model_name，实际应作为参数传入
            # 原有 ResNet 代码...
            if layer == 18:
                self.encoder = models.resnet18(pretrained=pretrained)
                encoder_dim = 512
            elif layer == 34:
                self.encoder = models.resnet34(pretrained=pretrained)
                encoder_dim = 512
            elif layer == 50:
                self.encoder = models.resnet50(pretrained=pretrained)
                encoder_dim = 2048
            else:
                self.encoder = models.resnet101(pretrained=pretrained)
                encoder_dim = 2048
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        elif model_name == 'Densenet':
            if layer == 121:
                self.encoder = models.densenet121(pretrained=pretrained)
                encoder_dim = 1024
            elif layer == 169:
                self.encoder = models.densenet169(pretrained=pretrained)
                encoder_dim = 1664
            elif layer == 201:
                self.encoder = models.densenet201(pretrained=pretrained)
                encoder_dim = 1920
            else:
                # 默认使用 DenseNet169
                self.encoder = models.densenet169(pretrained=pretrained)
                encoder_dim = 1664
            # DenseNet 的特征提取部分在 .features 中，不包含 avgpool 和分类器
            self.encoder = self.encoder.features
        
       # 特征投影层（统一处理全局池化 + 线性投影）
        self.feature_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # 将任意尺寸的特征图池化为 1x1
            nn.Flatten(),
            nn.Linear(encoder_dim, 256),
            nn.SiLU()
        )
        
        # 时间步编码器
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )
        
        # 噪声预测网络 (UNet风格的架构)
        input_dim = 6 + 256 + 256  # 噪声力(6) + 时间编码(256) + 图像特征(256)
        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 6)  # 输出: 预测的噪声
        )
        
    
    def forward(self, noisy_forces, t, images):
        """
        前向传播
        Args:
            noisy_forces: 加噪后的力值 [batch_size, 6]
            t: 时间步 [batch_size]
            images: 输入图像 [batch_size, channels, height, width]
        Returns:
            predicted_noise: 预测的噪声 [batch_size, 6]
        """
        # 提取图像特征
        image_features = self.encoder(images)
        image_features = self.feature_proj(image_features)  # [batch_size, 256]
        
        # 时间步编码
        t_embed = self.time_embed(t.unsqueeze(1).float())  # [batch_size, 256]
        
        # 拼接所有特征
        combined_features = torch.cat([noisy_forces, t_embed, image_features], dim=1)
        
        # 预测噪声
        predicted_noise = self.noise_predictor(combined_features)
        
        return predicted_noise


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calculate_accuracy(predict_wrench, label_wrench):
    accuracy = torch.sum(torch.abs(predict_wrench - label_wrench) / (torch.abs(label_wrench) + 1e-8), dim=0)
    return accuracy


class ForceEstimationDM:
    def __init__(self, cfg):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", default=None, type=str)
        parser.add_argument("--model_layer", default=None, type=str)
        parser.add_argument("--optimizer", default=None, type=str)
        parser.add_argument("--lrs", default=None, type=str2bool)
        parser.add_argument("--image_type", default=None, type=str)
        parser.add_argument("--cuda_index", default=None, type=int)
        parser.add_argument("--train_mode", default=None, type=str2bool)
        parser.add_argument("--test_object", default=None, type=str2bool)
        parser.add_argument("--mixed_image", default=None, type=str2bool)
        parser.add_argument("--pretrained", default=True, type=str2bool)
        parser.add_argument("--batch_size", default=None, type=int)
        parser.add_argument("--num_epoch", default=None, type=int)
        parser.add_argument("--learning_rate", default=None, type=float)
        parser.add_argument("--weight_decay", default=None, type=float)
        parser.add_argument("--end_factor", default=0.4, type=float)
        parser.add_argument("--total_iters", default=20, type=int)
        parser.add_argument("--num_workers", default=0, type=int, help="number of data loading workers")
        parser.add_argument("--num_timesteps", default=1000, type=int, help="number of diffusion timesteps")
        parser.add_argument("--beta_schedule", default="cosine", type=str, help="beta schedule type: linear, cosine")
        args = parser.parse_args()

        # parameters
        model_name = args.model_name if args.model_name is not None else cfg['model_list'][cfg['model_choice']][0]
        model_layer = args.model_layer if args.model_name is not None else cfg['model_list'][cfg['model_choice']][1]
        model_type = model_name + '-' + str(model_layer) + '-DM'
        optimizer = args.optimizer if args.optimizer is not None else 'Adam'
        self.lrs = args.lrs if args.lrs is not None else True
        image_type = args.image_type if args.image_type is not None else cfg['image_type']
        cuda_index = args.cuda_index if args.cuda_index is not None else cfg['cuda_index']
        train_mode = args.train_mode if args.train_mode is not None else cfg['train_mode']
        test_object = args.test_object if args.test_object is not None else cfg['test_object']
        mixed_image = args.mixed_image if args.mixed_image is not None else cfg['mixed_image']
        pretrained = args.pretrained & train_mode
        batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
        eval_batch = batch_size if train_mode else cfg['eval_batch']
        self.num_timesteps = args.num_timesteps
        self.beta_schedule = args.beta_schedule

        # model configuration
        self.device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
        self.model = DiffusionForceModel(model_name=model_name, layer=int(model_layer), pretrained=pretrained).to(self.device)
        self.lossFunction = nn.MSELoss(reduction='sum')  # 扩散模型使用MSE损失，与原代码保持一致使用sum

        # 初始化力值范围参数
        self.min_wrench = torch.tensor(min_wrench, dtype=torch.float).to(self.device)
        self.wrench_span = torch.tensor(wrench_span, dtype=torch.float).to(self.device)

        # 扩散模型参数
        self._setup_diffusion_parameters()

        # test dataset
        test_dataset = DTactDataset(mode='test', root_path=cfg['data_dir'], image_type=image_type,
                                    test_object=test_object, mixed_image=mixed_image)
        self.test_img_num = len(test_dataset)
        self.test_dataLoader = DataLoader(test_dataset, batch_size=eval_batch, shuffle=False, num_workers=args.num_workers)

        if train_mode:
            self.num_epoch = args.num_epoch if args.num_epoch is not None else cfg['num_epoch']
            learning_rate = args.learning_rate if args.learning_rate is not None else cfg['learning_rate']
            regularization = cfg['regularization']
            weight_decay = .0
            if args.weight_decay is not None:
                regularization = True
                weight_decay = args.weight_decay
            elif cfg['regularization']:
                weight_decay = cfg['weight_decay']

            # create directories
            time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
            self.save_dir = cfg['save_dir'] + "/" + model_type + time_suffix
            self.log_dir = self.save_dir + '/log'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # save basic information in the txt file
            self.write_basic_information(model_type, batch_size, self.num_epoch, learning_rate, regularization,
                                         weight_decay)

            # training dataset
            train_dataset = DTactDataset(mode='train', root_path=cfg['data_dir'],
                                         image_type=image_type, test_object=test_object, mixed_image=mixed_image)
            self.train_img_num = len(train_dataset)
            self.train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

            # optimizer
            if optimizer == 'SGD':
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
                                                   end_factor=args.end_factor, total_iters=args.total_iters)

            # wandb
            wandb_config = {
                "model_type": model_type,
                "image_type": image_type,
                "optimizer": optimizer,
                "test_object": test_object,
                "mixed_image": mixed_image,
                "pretrained": pretrained,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epoch": self.num_epoch,
                "weight_decay": weight_decay,
                "lrs": self.lrs,
                "end_factor": args.end_factor,
                "total_iters": args.total_iters,
                "num_timesteps": self.num_timesteps,
                "beta_schedule": self.beta_schedule
            }
            run_name = model_type + '_' + optimizer + '_lr-' + f'{learning_rate:.4f}' + '_OBJ-' + \
                       str(bool(test_object))[0] + '_MIX-' + str(bool(mixed_image))[0]
            if use_wandb:
                import wandb
                self.run_log = wandb.init(project="Force_Estimation", config=wandb_config, name=run_name)
            self.train()
        else:
            self.save_dir = cfg['save_dir'] + '/' + model_type + str(cfg['weights'][cfg['model_choice']][0])
            weights_path = self.save_dir + '/epoch_' + str(cfg['weights'][cfg['model_choice']][1]) + '.pt'
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.test_record(cfg['weights'][cfg['model_choice']][1])

    def _setup_diffusion_parameters(self):
        """设置扩散模型参数"""
        if self.beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(self.num_timesteps)
        else:
            self.betas = self._linear_beta_schedule(self.num_timesteps)
        
        # 确保参数在正确的设备上
        self.betas = self.betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 钳制避免浮点下溢
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, max=1.0 - 1e-7)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 预计算采样参数
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # 参数已在设备上（因为从已移动到设备的betas计算而来）
        # 无需再次to(device)

    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """线性噪声调度"""
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦噪声调度"""
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _forward_diffusion(self, x_0, t):
        """前向扩散过程 - 对力值加噪"""
        noise = torch.randn_like(x_0)
        
        # 确保索引在有效范围内
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def _sample_step(self, x, t, images):
        """单步采样"""
        with torch.no_grad():
            # 预测噪声
            noise_pred = self.model(x, t, images)
            
            # 计算去噪后的力值
            alpha_t = self.alphas[t].view(-1, 1)
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
            alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1)
            
            # 添加 epsilon 防止除零
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod_t + 1e-8)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod) * noise_pred)

            # 对方差也做保护
            variance = self.posterior_variance[t].view(-1, 1)
            variance = torch.clamp(variance, min=1e-8)

            # DDPM采样公式
            # 因为t的所有元素相同，取第一个判断即可
            if t[0] > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            
            # 采样
            x_prev = mean + torch.sqrt(variance) * z
            
            return x_prev

    def write_basic_information(self, model_type, batch_size, num_epoch, learning_rate, regularization,
                                weight_decay):
        basic_txt_path = self.save_dir + "/basic.txt"
        basic_txt_file = open(basic_txt_path, 'a')
        basic_txt_file.write(f"model: {model_type}\n")
        basic_txt_file.write(f"batch_size: {batch_size}\n")
        basic_txt_file.write(f"epoch: {num_epoch}\n")
        basic_txt_file.write(f"learning_rate: {learning_rate}\n")
        basic_txt_file.write(f"regularization: {regularization}\n")
        basic_txt_file.write(f"weight_decay: {weight_decay}\n")
        basic_txt_file.write(f"num_timesteps: {self.num_timesteps}\n")
        basic_txt_file.write(f"beta_schedule: {self.beta_schedule}\n")
        basic_txt_file.close()

    def train(self):
        start_time = time.time()
        test_best_loss = float('inf')
        iter_num = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        train_txt_path = self.save_dir + "/train.txt"
        train_txt_file = open(train_txt_path, 'a')
        best_epoch = 0
        
        for epoch in range(self.num_epoch):
            train_loss = 0.0
            train_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
            force_train_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
            self.model.train()
            
            for batch_idx, dataset in enumerate(self.train_dataLoader):
                self.optimizer.zero_grad()
                # 训练数据加载器只返回两个值：images和label_forces
                if len(dataset) == 2:
                    images, label_forces = dataset
                else:
                    images, label_forces, _ = dataset
                images = images.type(torch.FloatTensor)
                images = images.to(self.device)
                label_forces = label_forces.type(torch.FloatTensor)
                label_forces = label_forces.to(self.device)
                
                # 随机选择时间步（从1开始，避免t=0导致无噪声）
                t = torch.randint(1, self.num_timesteps, (label_forces.shape[0],), device=self.device)
                
                # 前向扩散 - 对真实力值加噪
                noisy_forces, noise = self._forward_diffusion(label_forces, t)
                
                # 预测噪声
                noise_pred = self.model(noisy_forces, t, images)
                
                # 计算噪声预测损失
                loss = self.lossFunction(noise_pred, noise)
                loss.backward()
                self.optimizer.step()
                
                # 计算去噪后的力值用于评估
                with torch.no_grad():
                    # 使用预测的噪声进行一步去噪
                    alpha_t = self.alphas[t].view(-1, 1)
                    alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
                    
                    sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1)
                    sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
                    predict_forces = (noisy_forces - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
                    
                    # 限制范围
                    predict_forces[predict_forces < 0] = 0
                    predict_forces[predict_forces > 1] = 1
                    
                    train_error += calculate_accuracy(predict_forces, label_forces)
                    force_train_error += torch.sum(torch.abs(predict_forces - label_forces), dim=0)
                
                train_loss += loss.item()
                iter_num += 1
            
            # learning rate schedule
            lr = self.optimizer.param_groups[0]["lr"]
            if self.lrs:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]

            train_loss /= self.train_img_num  # 与原代码保持一致，按样本数平均
            train_error = torch.div(train_error, self.train_img_num)
            force_train_error = torch.div(force_train_error, self.train_img_num)

            # test the model
            time_dif = get_time_dif(start_time)
            test_loss, force_test_error, force_test_error_span, test_error = self.test()
            if test_loss < test_best_loss:
                test_best_loss = test_loss
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = '******'
                best_epoch = epoch + 1
            else:
                improve = ''
            msg = '[Epoch: {}/{}, iter: {}], Lr: {:.4f},Train Loss: {:.3f}, Train Error: {}, {} '\
                  'Test Loss: {:.3f}, Test Error: {}, {}, Time: {} {}'
            print(msg.format(epoch + 1, self.num_epoch, iter_num, lr, train_loss,
                             np.round(force_train_error.detach().cpu().numpy(), 4),
                             np.round(train_error.detach().cpu().numpy(), 4),
                             test_loss, np.round(force_test_error.detach().cpu().numpy(), 4),
                             np.round(force_test_error_span.detach().cpu().numpy(), 3), time_dif / 60, improve))
            writer.add_scalar("loss/train", train_loss, iter_num)
            writer.add_scalar("loss/dev", test_loss, iter_num)
            train_txt_file.write(
                msg.format(epoch + 1, self.num_epoch, iter_num, lr, train_loss,
                           np.around(force_train_error.detach().cpu().numpy(), decimals=2),
                           np.round(train_error.detach().cpu().numpy(), 4),
                           test_loss, np.around(force_test_error.detach().cpu().numpy(), decimals=2),
                           np.round(force_test_error_span.detach().cpu().numpy(), 3), time_dif / 60, improve) + "\n")
            if use_wandb:
                log_dict = {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "lr": lr,
                    "epoch": epoch + 1, 
                    "train_error_x": force_train_error[0].item(),
                    "train_error_y": force_train_error[1].item(),
                    "train_error_z": force_train_error[2].item(),
                    "train_error_tx": force_train_error[3].item(),
                    "train_error_ty": force_train_error[4].item(),
                    "train_error_tz": force_train_error[5].item(),
                    "test_error_x": force_test_error[0].item(),
                    "test_error_y": force_test_error[1].item(),
                    "test_error_z": force_test_error[2].item(),
                    "test_error_tx": force_test_error[3].item(),
                    "test_error_ty": force_test_error[4].item(),
                    "test_error_tz": force_test_error[5].item(),
                }
                self.run_log.log(log_dict)
            self.model.train()
        writer.close()
        if use_wandb:
            self.run_log.finish()
        self.test_record(best_epoch)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
            force_test_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
            
            for batch_idx, dataset in enumerate(self.test_dataLoader):
                # 测试数据加载器返回三个值：images, label_forces, image_names
                if len(dataset) == 3:
                    images, label_forces, image_names = dataset
                else:
                    images, label_forces = dataset
                images = images.type(torch.FloatTensor)
                images = images.to(self.device)
                label_forces = label_forces.type(torch.FloatTensor)
                label_forces = label_forces.to(self.device)
                
                # 从纯噪声开始采样
                batch_size = images.shape[0]
                x = torch.randn_like(label_forces)  # 从标准正态分布开始
                
                # 逆向采样过程（从num_timesteps-1到1，t=0时直接输出均值）
                for t in reversed(range(1, self.num_timesteps)):
                    t_tensor = torch.tensor([t], device=self.device).repeat(batch_size)
                    x = self._sample_step(x, t_tensor, images)
                
                # 限制范围
                x[x < 0] = 0
                x[x > 1] = 1
                
                predict_forces = x
                # 使用力值的MSE损失进行评估（使用sum与训练保持一致）
                loss = nn.MSELoss(reduction='sum')(predict_forces, label_forces)
                test_loss += loss.item()
                test_error += calculate_accuracy(predict_forces, label_forces)
                force_test_error += torch.sum(torch.abs(predict_forces - label_forces), dim=0)
            
            test_loss /= self.test_img_num  # 与原代码保持一致，按样本数平均
            test_error = torch.div(test_error, self.test_img_num)
            force_test_error = torch.div(force_test_error, self.test_img_num)
            force_test_error_span = torch.mul(force_test_error, self.wrench_span)
        return test_loss, force_test_error, force_test_error_span, test_error

    def test_record(self, epoch):
        """测试记录函数 - 与原有逻辑保持一致"""
        self.model.eval()
        test_txt_path = self.save_dir + "/test_" + str(epoch) + '.txt'
        if os.path.exists(test_txt_path):
            os.remove(test_txt_path)
        test_txt_file = open(test_txt_path, 'a')
        test_image_dir = self.save_dir + "/image_" + str(epoch)
        if not os.path.exists(test_image_dir):
            os.makedirs(test_image_dir)
        image_name_list = []
        mean_error_list = []
        truth_list = []
        predict_list = []
        error_list = []
        img_number = 0
        
        with torch.no_grad():
            start_time = time.time()
            for batch_idx, dataset in enumerate(self.test_dataLoader):
                # 测试数据加载器返回三个值：images, label_forces, image_paths
                if len(dataset) == 3:
                    images, label_forces, image_paths = dataset
                else:
                    images, label_forces = dataset
                    image_paths = [f"unknown_{i}" for i in range(len(images))]
                images = images.type(torch.FloatTensor)
                images = images.to(self.device)
                label_forces = label_forces.type(torch.FloatTensor)
                label_forces = label_forces.to(self.device)
                
                # 从纯噪声开始采样
                batch_size = images.shape[0]
                predict_forces = torch.randn_like(label_forces)
                
                # 逆向采样过程（从num_timesteps-1到1，t=0时直接输出均值）
                for t in reversed(range(1, self.num_timesteps)):
                    t_tensor = torch.tensor([t], device=self.device).repeat(batch_size)
                    predict_forces = self._sample_step(predict_forces, t_tensor, images)
                
                # 限制范围
                predict_forces[predict_forces < 0] = 0
                predict_forces[predict_forces > 1] = 1
                
                force_truth = torch.add(torch.mul(label_forces, self.wrench_span), self.min_wrench)
                force_predict = torch.add(torch.mul(predict_forces, self.wrench_span), self.min_wrench)
                force_error = torch.abs(force_truth - force_predict)

                for i in range(images.size(0)):
                    msg = "{}: {} Mean: {:.4f} Tru: {} Pre: {} Error: {}" \
                        .format(1 + i + img_number * batch_idx, image_paths[i].split('image')[-1],
                                force_error[i].mean(),
                                np.round(force_truth[i].detach().cpu().numpy(), 4),
                                np.round(force_predict[i].detach().cpu().numpy(), 4),
                                np.round(force_error[i].detach().cpu().numpy(), 4))
                    test_txt_file.write(msg + "\n")
                    image_name_list.append(image_paths[i])
                    mean_error_list.append(np.round(force_error[i].mean().item(), 4))
                    truth_list.append(np.round(force_truth[i].detach().cpu().numpy(), 4))
                    predict_list.append(np.round(force_predict[i].detach().cpu().numpy(), 4))
                    error_list.append(np.round(force_error[i].detach().cpu().numpy(), 4))
                img_number = images.size(0)
        
        test_txt_file.write("--------------------------------------------\n")
        time_dif = get_time_dif(start_time)
        final_error = np.mean(error_list, axis=0)
        msg_time = "Final Error: {}, Time: {:2f}, FPS: {:2f} \n"\
            .format(final_error, time_dif, self.test_img_num / time_dif)
        print(msg_time)
        test_txt_file.write(msg_time)
        test_txt_file.write("--------------------------------------------\n")
        sorted_index = np.argsort(np.array(mean_error_list))
        for j in range(len(error_list)):
            msg_sort = "{}: {} Tru: {} Pre: {}  Err: {}" \
                .format(1 + j, image_name_list[sorted_index[j]].split('image')[-1],
                        truth_list[sorted_index[j]],
                        predict_list[sorted_index[j]],
                        error_list[sorted_index[j]])
            test_txt_file.write(msg_sort + "\n")
            saved_img = cv2.imread(image_name_list[sorted_index[j]])
            cv2.imwrite(test_image_dir + '/' + str(j + 1) + '.png', saved_img)

        truth_list = np.array(truth_list)
        predict_list = np.array(predict_list)
        print(truth_list.shape)
        print(predict_list.shape)
        np.save(self.save_dir + '/image_' + str(epoch) + '/truth_list.npy', truth_list)
        np.save(self.save_dir + '/image_' + str(epoch) + '/predict_list.npy', predict_list)


if __name__ == '__main__':
    f = open("force_config.yaml", 'r+', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)
    force_estimation = ForceEstimationDM(config)