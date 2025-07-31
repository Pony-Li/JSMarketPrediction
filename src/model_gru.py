import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning
from pytorch_lightning import LightningModule
from utils import calculate_r2

# 自定义超参数类
class Hyperparameters:
    def __init__(self):
        self.use_gpu = True  # 是否使用GPU
        self.gpu_id = 0  # GPU ID
        self.seed = 2024  # 随机种子
        self.model_type = 'gru'  # 模型类型
        self.num_workers = 0  # 数据加载时的线程数
        self.batch_size = 1  # 批量大小
        self.learning_rate = 1e-3  # 学习率
        self.weight_decay = 5e-5 # 权重衰减
        self.dropout_rate = 0.1  # Dropout率
        self.hidden_layer_size = 512  # 隐藏层大小
        self.early_stopping_patience = 10  # 早停耐心值
        self.max_epochs = 50  # 最大训练轮数

params = Hyperparameters()
pytorch_lightning.seed_everything(params.seed)


# 模型定义
class GRUNetworkWithConv(LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, learning_rate=1e-4, weight_decay=0.1):
        super().__init__()
        self.save_hyperparameters()  # 保存超参数
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)  # GRU层
        self.layer_norm = nn.LayerNorm(hidden_size)  # 层归一化
        self.batch_norm = nn.BatchNorm1d(input_size)  # 批归一化
        self.gate = nn.Linear(hidden_size, hidden_size)  # 门控线性层
        self.dense = nn.Sequential(nn.Linear(hidden_size, output_size))  # 全连接层
        self.gate_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=2, groups=hidden_size)  # 1D卷积层
        self.prev_hidden_state = None  # 上一隐藏状态
        self.validation_step_outputs = []  # 验证步骤输出

    def forward(self, x):
        x = x.squeeze(0)  # 去除多余的维度
        batch_size, timestep, input_size = x.size()  # 获取输入形状
        h0 = self._initialize_hidden_state(batch_size, x.device)  # 初始化隐藏状态
        gru_outputs, hn = self.gru(x, h0)  # GRU前向传播
        self.prev_hidden_state = hn.detach()  # 更新隐藏状态
        output = gru_outputs.permute(0, 2, 1)  # 调整维度
        gating_values = torch.sigmoid(self.gate_conv(output))[:, :, :output.size(-1)]  # 计算门控值
        output = output * gating_values  # 应用门控
        output = output.permute(0, 2, 1)  # 调整维度
        output = self.dense(output)  # 全连接层
        return output.squeeze(-1)  # 去除多余的维度

    def _initialize_hidden_state(self, batch_size, device):
        if self.prev_hidden_state is None:
            return torch.zeros(self.hparams.num_layers, batch_size, self.gru.hidden_size).to(device)  # 初始化隐藏状态
        else:
            prev_batch_size = self.prev_hidden_state.size(1)  # 获取上一隐藏状态的批量大小
            if prev_batch_size > batch_size:
                return self.prev_hidden_state[:, :batch_size, :]  # 截取隐藏状态
            elif prev_batch_size < batch_size:
                padding_size = batch_size - prev_batch_size  # 计算填充大小
                return torch.cat([self.prev_hidden_state, self.prev_hidden_state[:, -1:, :].repeat(1, padding_size, 1)], dim=1)  # 填充隐藏状态
            else:
                return self.prev_hidden_state  # 返回上一隐藏状态

    def training_step(self, batch,):
        x, y, w, mask = batch  # 获取输入数据
        y_hat = self(x)  # 模型预测
        if y_hat.shape != y.shape:
            y_hat = y_hat.view_as(y)  # 调整预测形状
            mask = mask.view_as(y)  # 调整掩码形状
        y_hat, y = y_hat[mask == 1], y[mask == 1]  # 应用掩码
        loss = F.mse_loss(y_hat, y, reduction='none').mean()  # 计算损失
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))  # 记录损失
        return loss

    def validation_step(self, batch,):
        x, y, w, mask = batch  # 获取输入数据
        y_hat = self(x)  # 模型预测
        if y_hat.shape != y.shape:
            y_hat = y_hat.view_as(y)  # 调整预测形状
        y_hat, y = y_hat[mask == 1], y[mask == 1]  # 应用掩码
        loss = F.mse_loss(y_hat, y, reduction='none').mean()  # 计算损失
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))  # 记录损失
        self.validation_step_outputs.append((y_hat, y, w, mask))  # 保存验证输出
        return loss

    def on_validation_epoch_end(self):
        """在验证轮次结束时计算验证R²分数"""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()  # 拼接真实值
        prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()  # 拼接预测值
        weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()  # 拼接权重
        masks =  torch.cat([x[3] for x in self.validation_step_outputs]).cpu().numpy()  # 拼接掩码

        if prob.shape != y.shape or prob.shape != masks.shape or prob.shape != weights.shape:
            prob = torch.tensor(prob)  # 转换为张量
            weights = torch.tensor(weights)  # 转换为张量
            masks = torch.tensor(masks)  # 转换为张量
            y = torch.tensor(y)  # 转换为张量
            prob = prob.view_as(y)  # 调整预测形状
            weights = weights.view_as(y)  # 调整权重形状
            masks = masks.view_as(y)  # 调整掩码形状
            y = y[masks==1]  # 应用掩码
            weights = weights[masks==1]  # 应用掩码
            prob = prob[masks==1]  # 应用掩码

        val_r_square = calculate_r2(y, prob, weights)  # 计算R²分数
        self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)  # 记录R²分数
        self.validation_step_outputs.clear()  # 清空验证输出

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)  # 定义优化器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # 定义学习率调度器
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}  # 返回优化器和调度器

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch  # 获取当前轮次
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}  # 获取指标
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}  # 格式化指标
        print(f"Epoch {epoch}: {formatted_metrics}")  # 打印指标
        self.prev_hidden_state = None  # 重置隐藏状态