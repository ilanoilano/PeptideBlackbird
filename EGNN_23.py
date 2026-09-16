#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGNN训练模块 (EGNN_23.py)
功能：分子向量化 + 端到端训练（合并EGNN_2和EGNN_3）

输入：
- egnn/raw/*.npz (train_data.npz, val_data.npz, test_data.npz)

处理逻辑：
1. 加载原子特征和坐标
2. 构建分子图（原子为节点，距离<5Å为边）
3. 多层EGNN图卷积（参数可训练）
4. 全局池化汇聚成分子向量
5. MLP回归输出预测结合能
6. 端到端训练（EGNN层 + MLP层参数一起更新）
7. 早停：验证集连续10轮不提升

输出：
- egnn/models/best_model.pt
- egnn/models/training_history.json
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent))

import config


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# 设备配置
# 不再强制禁用CUDA，让PyTorch自动检测
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 删除或注释掉这行

# 设备配置：自动检测CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


class EGNNEncoder(nn.Module):
    """
    EGNN编码器
    
    等变图神经网络层，保持旋转平移等变性
    """
    
    def __init__(self, in_features: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 输入投影
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # EGNN层
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, features, coords, edge_index):
        """
        Args:
            features: (n_atoms, in_features)
            coords: (n_atoms, 3)
            edge_index: (2, n_edges) 边索引
        
        Returns:
            h: (n_atoms, hidden_dim) 原子特征
            coords: (n_atoms, 3) 更新后的坐标
        """
        # 输入投影
        h = self.input_proj(features)
        
        # EGNN层
        for layer in self.egnn_layers:
            h, coords = layer(h, coords, edge_index)
        
        return h, coords


class EGNNLayer(nn.Module):
    """
    单层EGNN
    
    参考: E(n) Equivariant Graph Neural Networks (Satorras et al., 2021)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 边特征网络 (消息函数)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 坐标更新网络
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # 节点更新网络
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, coords, edge_index):
        """
        Args:
            h: (n_atoms, hidden_dim)
            coords: (n_atoms, 3)
            edge_index: (2, n_edges)
        """
        row, col = edge_index
        
        # 计算边特征
        coord_diff = coords[row] - coords[col]  # (n_edges, 3)
        dist = torch.norm(coord_diff, dim=1, keepdim=True)  # (n_edges, 1)
        
        # 消息
        edge_input = torch.cat([h[row], h[col], dist], dim=-1)  # (n_edges, hidden_dim*2+1)
        message = self.edge_mlp(edge_input)  # (n_edges, hidden_dim)
        
        # 坐标更新 (保持等变性)
        coord_weight = self.coord_mlp(message)  # (n_edges, 1)
        coord_update = coord_weight * coord_diff  # (n_edges, 3)
        
        # 聚合坐标更新
        coords_new = coords.clone()
        for i in range(coords.shape[0]):
            mask = row == i
            if mask.sum() > 0:
                coords_new[i] += coord_update[mask].sum(dim=0)
        
        # 聚合消息
        h_agg = torch.zeros_like(h)
        for i in range(h.shape[0]):
            mask = col == i
            if mask.sum() > 0:
                h_agg[i] = message[mask].sum(dim=0)
        
        # 节点更新
        h_new = self.node_mlp(torch.cat([h, h_agg], dim=-1))
        h_new = h + h_new  # 残差连接
        
        return h_new, coords_new


class EGNNModel(nn.Module):
    """
    完整EGNN模型
    
    EGNN编码器 + 全局池化 + MLP回归
    """
    
    def __init__(self, in_features: int = 20, hidden_dim: int = 128, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # EGNN编码器
        self.encoder = EGNNEncoder(in_features, hidden_dim, num_layers)
        
        # MLP回归头
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features, coords, edge_index, batch):
        """
        Args:
            features: (total_atoms, in_features)
            coords: (total_atoms, 3)
            edge_index: (2, n_edges)
            batch: (total_atoms,) 每个原子属于哪个分子
        
        Returns:
            energy: (n_molecules,) 预测结合能
        """
        # EGNN编码
        h, coords = self.encoder(features, coords, edge_index)
        
        # 全局平均池化
        # 对每个分子分别池化
        unique_batches = torch.unique(batch)
        h_mol = []
        for b in unique_batches:
            mask = batch == b
            h_mol.append(h[mask].mean(dim=0))  # 平均池化
        h_mol = torch.stack(h_mol)
        
        # MLP回归
        energy = self.mlp(h_mol).squeeze(-1)
        
        return energy


def build_graph(features, coords, cutoff=5.0):
    """
    从坐标构建图（边索引）
    
    Args:
        features: (n_atoms, n_features)
        coords: (n_atoms, 3)
        cutoff: 距离阈值（Å）
    
    Returns:
        edge_index: (2, n_edges)
    """
    n_atoms = len(coords)
    
    # 计算距离矩阵
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    dist_matrix = torch.cdist(coords_tensor, coords_tensor)
    
    # 找到距离小于cutoff的边
    edge_index = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j and dist_matrix[i, j] < cutoff:
                edge_index.append([i, j])
    
    if len(edge_index) == 0:
        # 如果没有边，添加自环
        edge_index = [[i, i] for i in range(n_atoms)]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    return edge_index


def collate_graphs(batch_data):
    """
    将多个分子数据组合成一个batch
    """
    all_features = []
    all_coords = []
    all_edge_indices = []
    batch_indices = []
    energies = []

    atom_offset = 0

    for i, (features, coords, energy) in enumerate(batch_data):
        # 确保 features 和 coords 是 numpy 数组
        if isinstance(features, np.ndarray) and features.dtype == object:
            features = np.array(features.tolist(), dtype=np.float32)
        if isinstance(coords, np.ndarray) and coords.dtype == object:
            coords = np.array(coords.tolist(), dtype=np.float32)

        # 如果已经是 tensor，转换为 numpy
        if torch.is_tensor(features):
            features = features.numpy()
        if torch.is_tensor(coords):
            coords = coords.numpy()

        # 确保是 float32
        features = np.array(features, dtype=np.float32)
        coords = np.array(coords, dtype=np.float32)

        n_atoms = len(features)

        all_features.append(features)
        all_coords.append(coords)
        energies.append(energy)

        # 构建图
        edge_index = build_graph(features, coords)
        edge_index = edge_index + atom_offset
        all_edge_indices.append(edge_index)

        batch_indices.extend([i] * n_atoms)
        atom_offset += n_atoms

    # 拼接
    features = torch.tensor(np.concatenate(all_features, axis=0), dtype=torch.float32)
    coords = torch.tensor(np.concatenate(all_coords, axis=0), dtype=torch.float32)
    edge_index = torch.cat(all_edge_indices, dim=1)
    batch = torch.tensor(batch_indices, dtype=torch.long)
    energies = torch.tensor(energies, dtype=torch.float32)

    return features, coords, edge_index, batch, energies


def load_npz_data(data_file: Path):
    """加载npz数据"""
    data = np.load(data_file, allow_pickle=True)
    
    features_list = data['features']
    coords_list = data['coords']
    energies = data['energies']
    
    dataset = []
    for i in range(len(energies)):
        dataset.append((features_list[i], coords_list[i], energies[i]))
    
    return dataset


def train_epoch(model, train_loader, optimizer, criterion):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_data in train_loader:
        features, coords, edge_index, batch, energies = collate_graphs(batch_data)
        
        features = features.to(DEVICE)
        coords = coords.to(DEVICE)
        edge_index = edge_index.to(DEVICE)
        batch = batch.to(DEVICE)
        energies = energies.to(DEVICE)
        
        # 前向传播
        pred_energies = model(features, coords, edge_index, batch)
        
        # 计算损失
        loss = criterion(pred_energies, energies)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            features, coords, edge_index, batch, energies = collate_graphs(batch_data)
            
            features = features.to(DEVICE)
            coords = coords.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            batch = batch.to(DEVICE)
            energies = energies.to(DEVICE)
            
            pred_energies = model(features, coords, edge_index, batch)
            loss = criterion(pred_energies, energies)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main(data_dir: Optional[Path] = None,
         output_dir: Optional[Path] = None,
         hidden_dim: int = None,
         num_layers: int = None,
         num_epochs: int = None,
         batch_size: int = None,
         lr: float = None,
         patience: int = None,
         C: float = None,
         target_name: Optional[str] = None):
    """
    主训练函数

    Args:
        target_name: 靶点名称（用于确定EGNN目录）
    """
    # 使用target_name确定EGNN目录
    C=config.lr_recession
    if target_name is not None:
        egnn_dirs = config.get_egnn_dirs(target_name)
        if data_dir is None:
            data_dir = egnn_dirs["raw"]
        if output_dir is None:
            output_dir = egnn_dirs["models"]
    else:
        # fallback：使用原默认路径
        if data_dir is None:
            data_dir = config.BASE_DIR / "egnn" / "raw"
        if output_dir is None:
            output_dir = config.BASE_DIR / "egnn" / "models"

    # ... 其余代码保持不变
    
    set_seed(42)
    
    # 路径
    if data_dir is None:
        data_dir = config.BASE_DIR / "egnn" / "raw"
    if output_dir is None:
        output_dir = config.BASE_DIR / "egnn" / "models"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("EGNN训练")
    print(f"隐藏维度: {hidden_dim}")
    print(f"层数: {num_layers}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"早停耐心: {patience}")
    
    # 加载数据
    print(f"\n[1/3] 加载数据...")
    train_data = load_npz_data(data_dir / "train_data.npz")
    val_data = load_npz_data(data_dir / "val_data.npz")
    
    print(f"  训练集: {len(train_data)} 个样本")
    print(f"  验证集: {len(val_data)} 个样本")
    
    if len(train_data) == 0:
        print("错误: 训练集为空")
        return
    
    # 创建数据加载器
    train_loader = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
    val_loader = [val_data[i:i+batch_size] for i in range(0, len(val_data), batch_size)]
    
    # 创建模型
    print(f"\n[2/3] 创建模型...")
    model = EGNNModel(in_features=20, hidden_dim=hidden_dim, num_layers=num_layers)
    model = model.to(DEVICE)
    try:
        model = torch.compile(model)
        print("  ✓ 模型已编译 (torch.compile)")
    except:
        print("  ⚠️ torch.compile 不可用，跳过")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {n_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)  # Huber损失对异常值更鲁棒
    
    # 训练循环
    # 训练循环
    print(f"\n[3/3] 开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    base_lr = lr  # 保存初始基准学习率

    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        # 验证
        val_loss = validate(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history['lr'].append(current_lr)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.2e}")

        # ---------------- 自定义学习率调度逻辑 ----------------
        if val_loss < best_val_loss:
            # 验证集性能改善：更新最优，学习率恢复原始base_lr
            best_val_loss = val_loss
            patience_counter = 0
            # 恢复为初始学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr
            print(f"  ✓ 新最优val_loss，学习率恢复初始值 {base_lr:.2e}")
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / "best_model.pt")
            print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            # 验证集没有改善：学习率减半
            patience_counter += 1
            new_lr = current_lr * C
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"  ⚠️ val_loss未改善，学习率降低 → {new_lr:.2e}")

            if patience_counter >= patience:
                print(f"\n早停触发！连续{patience}轮验证损失未提升")
                break

    # 保存训练历史（增加lr记录，方便绘图查看lr变化）
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存至: {output_dir / 'best_model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    # 从config读取默认值
    default_hidden_dim = config.EGNN_CONFIG.get("hidden_dim", 128)
    default_num_layers = config.EGNN_CONFIG.get("num_layers", 4)
    default_num_epochs = config.EGNN_CONFIG.get("num_epochs", 100)
    default_batch_size = config.EGNN_CONFIG.get("batch_size", 8)
    default_lr = config.EGNN_CONFIG.get("learning_rate", 1e-3)
    default_patience = config.EGNN_CONFIG.get("patience", 10)

    parser = argparse.ArgumentParser(description='EGNN训练')
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--hidden-dim', type=int, default=default_hidden_dim)
    parser.add_argument('--num-layers', type=int, default=default_num_layers)
    parser.add_argument('--num-epochs', type=int, default=default_num_epochs)
    parser.add_argument('--batch-size', type=int, default=default_batch_size)
    parser.add_argument('--lr', type=float, default=default_lr)
    parser.add_argument('--patience', type=int, default=default_patience)
    # 新增下面一行
    parser.add_argument('--target', type=str, default=None, help="靶点名称")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        target_name=args.target  # 传入target_name
    )