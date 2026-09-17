#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGNN预测器 (egnn_predictor.py)
功能：加载训练好的EGNN模型并进行预测

输入：
- pdbqt_path: PDBQT文件路径
- model_path: EGNN模型路径（可选，默认使用config中的路径）

输出：
- 预测结合能（kcal/mol）

流程：
1. 从PDBQT解析原子坐标和元素类型
2. 提取原子特征（元素one-hot、杂化方式等）
3. 构建分子图
4. EGNN预测
"""

import sys
import random
import re
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List

# 强制使用CPU，RTX 5060暂不支持CUDA
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, str(Path(__file__).parent))

import config
from EGNN_23 import EGNNModel, build_graph


# 元素类型 one-hot (10维)
ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'other']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}

# 原子特征维度
N_FEATURES = 20


def parse_pdbqt(pdbqt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    解析PDBQT文件，提取原子坐标和元素类型
    
    Args:
        pdbqt_path: PDBQT文件路径
    
    Returns:
        coords: (n_atoms, 3) 原子坐标
        elements: (n_atoms,) 元素类型列表
    """
    coords = []
    elements = []
    
    with open(pdbqt_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # 解析坐标 (列30-38, 38-46, 46-54)
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    
                    # 解析元素（列77-78，或从原子名推断）
                    element = line[76:78].strip()
                    if not element:
                        # 从原子名推断（列12-16）
                        atom_name = line[12:16].strip()
                        if atom_name:
                            element = atom_name[0]
                    
                    elements.append(element.upper())
                    
                except (ValueError, IndexError):
                    continue
    
    if not coords:
        raise ValueError(f"PDBQT文件中没有原子: {pdbqt_path}")
    
    return np.array(coords), elements


def extract_atom_features(coords: np.ndarray, elements: List[str]) -> np.ndarray:
    """
    提取原子特征
    
    Args:
        coords: (n_atoms, 3) 原子坐标
        elements: (n_atoms,) 元素类型列表
    
    Returns:
        features: (n_atoms, N_FEATURES) 原子特征矩阵
    """
    n_atoms = len(coords)
    features = np.zeros((n_atoms, N_FEATURES))
    
    for i, element in enumerate(elements):
        # 1. 元素类型 one-hot (10维)
        elem_idx = ELEMENT_TO_IDX.get(element, ELEMENT_TO_IDX['other'])
        features[i, elem_idx] = 1.0
        
        # 2. 杂化方式 (4维) - 简化处理，基于元素推断
        # 这里使用简化规则，实际应该用RDKit
        if element in ['C']:
            features[i, 10] = 1.0  # SP3
        elif element in ['N', 'O']:
            features[i, 11] = 1.0  # SP2
        else:
            features[i, 13] = 1.0  # other
        
        # 3. 形式电荷 (1维) - 简化，设为0
        features[i, 14] = 0.0
        
        # 4. 是否是氢键供体 (1维)
        features[i, 15] = 1.0 if element in ['N', 'O'] else 0.0
        
        # 5. 是否是氢键受体 (1维)
        features[i, 16] = 1.0 if element in ['N', 'O'] else 0.0
        
        # 6. 是否是芳香族 (1维) - 简化
        features[i, 17] = 0.0
        
        # 7. 原子度数 (1维，归一化）- 简化
        features[i, 18] = 0.5
        
        # 8. 原子质量 (1维，归一化) - 简化
        mass_map = {'C': 12, 'N': 14, 'O': 16, 'S': 32, 'P': 31, 'H': 1}
        mass = mass_map.get(element, 12)
        features[i, 19] = mass / 100.0
    
    return features


class EGNNPredictor:
    """
    EGNN预测器
    
    加载训练好的EGNN模型，对PDBQT文件进行预测
    """

    def __init__(self, model_path: Optional[Path] = None,
                 target_name: Optional[str] = None,
                 hidden_dim: int = None, num_layers: int = None):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径（默认从config读取）
            hidden_dim: 隐藏层维度（默认从config读取）
            num_layers: EGNN层数（默认从config读取）
        """
        if model_path is None:
            if target_name is not None:
                model_path = config.get_egnn_dirs(target_name)["models"] / "best_model.pt"
            else:
                model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt" 

        # 【修复】从 config 读取，而不是硬编码
        if hidden_dim is None:
            hidden_dim = config.EGNN_CONFIG.get("hidden_dim", 128)
        if num_layers is None:
            num_layers = config.EGNN_CONFIG.get("num_layers", 4)
        
        self.model_path = Path(model_path)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.device = torch.device('cpu')  # 强制CPU，RTX 5060暂不支持
        
        self._load_model()

    def _load_model(self):
        """加载EGNN模型"""
        print(f"[EGNNPredictor] 正在加载EGNN模型: {self.model_path}")

        if not self.model_path.exists():
            print(f"\n{'=' * 60}")
            print(f"【EGNN错误】模型文件不存在")
            print(f"{'=' * 60}")
            print(f"  期望路径: {self.model_path}")
            print(f"\n  可能原因:")
            print(f"    1. EGNN训练未完成，模型文件未生成")
            print(f"    2. 模型文件被删除或移动")
            print(f"    3. 路径配置错误")
            print(f"\n  解决方案:")
            print(f"    1. 运行EGNN数据准备:")
            print(f"       python EGNN_1.py")
            print(f"    2. 运行EGNN模型训练:")
            print(f"       python EGNN_23.py")
            print(f"    3. 检查模型路径: ls -la egnn/models/")
            print(f"{'=' * 60}\n")
            raise FileNotFoundError(f"EGNN模型不存在: {self.model_path}")

        try:
            # 创建模型
            print(f"[EGNNPredictor] 创建EGNN模型结构...")
            self.model = EGNNModel(
                in_features=N_FEATURES,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )

            # 加载权重
            print(f"[EGNNPredictor] 加载模型权重...")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 检查checkpoint内容
            if 'model_state_dict' not in checkpoint:
                print(f"\n{'=' * 60}")
                print(f"【EGNN错误】Checkpoint格式不正确")
                print(f"{'=' * 60}")
                print(f"  模型路径: {self.model_path}")
                print(f"  可用键: {list(checkpoint.keys())}")
                print(f"\n  可能原因:")
                print(f"    1. 模型文件损坏")
                print(f"    2. 使用不同版本的PyTorch保存/加载")
                print(f"    3. 训练过程中断，文件不完整")
                print(f"\n  解决方案:")
                print(f"    1. 删除损坏的模型文件并重新训练")
                print(f"    2. 检查PyTorch版本兼容性")
                print(f"    3. 重新运行: python EGNN_23.py")
                print(f"{'=' * 60}\n")
                raise KeyError("checkpoint中缺少'model_state_dict'")

            # ========== 修复：移除 _orig_mod. 前缀 ==========
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # 去掉 '_orig_mod.' 前缀
                else:
                    new_key = key
                new_state_dict[new_key] = value

            self.model.load_state_dict(new_state_dict)
            # ========== 修复结束 ==========

            self.model.to(self.device)
            self.model.eval()

            print(f"[EGNNPredictor] ✓ EGNN模型加载成功")
            print(f"[EGNNPredictor]   设备: {self.device}")
            print(f"[EGNNPredictor]   隐藏维度: {self.hidden_dim}")
            print(f"[EGNNPredictor]   层数: {self.num_layers}")

        except Exception as e:
            print(f"\n{'=' * 60}")
            print(f"【EGNN错误】模型加载失败")
            print(f"{'=' * 60}")
            print(f"  异常类型: {type(e).__name__}")
            print(f"  异常信息: {e}")
            print(f"  模型路径: {self.model_path}")
            print(f"\n  可能原因:")
            print(f"    1. PyTorch版本不兼容")
            print(f"    2. 模型文件损坏")
            print(f"    3. CUDA/CPU设备不匹配")
            print(f"\n  解决方案:")
            print(f"    1. 检查PyTorch版本: python -c 'import torch; print(torch.__version__)'")
            print(f"    2. 重新训练模型: python EGNN_23.py")
            print(f"    3. 检查CUDA可用性: python -c 'import torch; print(torch.cuda.is_available())'")
            print(f"{'=' * 60}\n")
            raise
    
    def predict(self, pdbqt_path: Path) -> float:
        """
        预测PDBQT文件的结合能
        
        Args:
            pdbqt_path: PDBQT文件路径
        
        """
        pdbqt_path = Path(pdbqt_path)
        
        print(f"[EGNNPredictor] 开始预测: {pdbqt_path.name}")
        
        if not pdbqt_path.exists():
            print(f"\n{'='*60}")
            print(f"【EGNN错误】PDBQT文件不存在")
            print(f"{'='*60}")
            print(f"  期望路径: {pdbqt_path}")
            print(f"\n  可能原因:")
            print(f"    1. 配体生成失败")
            print(f"    2. 文件路径错误")
            print(f"    3. 临时文件被清理")
            print(f"\n  解决方案:")
            print(f"    1. 检查 ligand_generator.py 是否正常工作")
            print(f"    2. 检查临时目录权限")
            print(f"    3. 重新生成分子")
            print(f"{'='*60}\n")
            raise FileNotFoundError(f"PDBQT文件不存在: {pdbqt_path}")
        
        try:
            # 1. 解析PDBQT
            print(f"[EGNNPredictor] 解析PDBQT文件...")
            coords, elements = parse_pdbqt(pdbqt_path)
            print(f"[EGNNPredictor]   原子数: {len(coords)}")
            
            # 2. 提取特征
            print(f"[EGNNPredictor] 提取原子特征...")
            features = extract_atom_features(coords, elements)
            print(f"[EGNNPredictor]   特征维度: {features.shape}")
            
            # 3. 构建图
            print(f"[EGNNPredictor] 构建分子图...")
            edge_index = build_graph(features, coords)
            print(f"[EGNNPredictor]   边数: {edge_index.shape[1]}")
            
            # 4. 转换为tensor
            print(f"[EGNNPredictor] 转换为tensor...")
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            coords_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)
            edge_index_tensor = edge_index.to(self.device)
            batch_tensor = torch.zeros(len(coords), dtype=torch.long).to(self.device)
            
            # 5. 预测
            print(f"[EGNNPredictor] EGNN前向传播...")
            with torch.no_grad():
                energy = self.model(features_tensor, coords_tensor, edge_index_tensor, batch_tensor)
                energy = energy.item()
            
            print(f"[EGNNPredictor] ✓ 预测完成: energy={energy:.4f} kcal/mol")
            return energy
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"【EGNN错误】预测失败")
            print(f"{'='*60}")
            print(f"  异常类型: {type(e).__name__}")
            print(f"  异常信息: {e}")
            print(f"  文件: {pdbqt_path}")
            print(f"\n  可能原因:")
            print(f"    1. PDBQT文件格式错误")
            print(f"    2. 原子数过多（内存不足）")
            print(f"    3. 模型输入维度不匹配")
            print(f"    4. CUDA/CPU设备错误")
            print(f"\n  解决方案:")
            print(f"    1. 检查PDBQT文件: cat {pdbqt_path} | head -20")
            print(f"    2. 检查原子数: grep -c '^ATOM' {pdbqt_path}")
            print(f"    3. 验证模型输入维度")
            print(f"    4. 检查设备可用性")
            print(f"{'='*60}\n")
            raise


def create_egnn_predictor(target_name: Optional[str] = None) -> EGNNPredictor:
    """
    创建EGNN预测器（如果模型存在）
    
    Args:
        target_name: 靶点名称。若提供，从 egnn/{target}/models/ 加载；
                    否则从 egnn/models/ 加载。
    """
    if target_name is not None:
        model_path = config.get_egnn_dirs(target_name)["models"] / "best_model.pt"
    else:
        model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"

    if not model_path.exists():
        print(f"【EGNN错误】模型文件不存在: {model_path}")
        print("  请先运行以下步骤训练模型:")
        print("    1. python EGNN_1.py  # 准备数据")
        print("    2. python EGNN_23.py  # 训练模型")
        raise FileNotFoundError(f"EGNN模型不存在: {model_path}")

    try:
        predictor = EGNNPredictor(model_path=model_path, target_name=target_name)
        print(f"【EGNN】模型加载成功: {model_path}")
        return predictor
    except Exception as e:
        print(f"【EGNN错误】模型加载失败: {e}")
        raise RuntimeError(f"无法加载EGNN模型: {e}") from e



def main():
    """测试代码"""
    print("="*60)
    print("EGNN预测器测试")
    print("="*60)
    
    # 检查模型是否存在
    model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"
    
    if not model_path.exists():
        print(f"\n模型不存在: {model_path}")
        print("请先运行EGNN训练")
        return
    
    # 创建预测器
    print("\n加载EGNN模型...")
    try:
        predictor = EGNNPredictor(model_path)
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    # 测试预测（需要一个PDBQT文件）
    # 这里只是示例，实际使用时需要提供真实的PDBQT文件
    print("\n注意: 需要提供PDBQT文件进行测试")
    print("="*60)


if __name__ == "__main__":
    main()
