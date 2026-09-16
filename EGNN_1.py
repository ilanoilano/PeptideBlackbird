#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGNN数据准备模块 (EGNN_1.py)
功能：序列 + Vina分数 → 原子特征 + 坐标 → 划分数据集

【改进】增量处理：已处理序列记录在 processed_sequences.txt，不删除源文件
支持断点续跑，反复运行不重复处理
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

sys.path.insert(0, str(Path(__file__).parent))

import config
from ligand_generator import build_peptide_with_rdkit, generate_3d_conformation, add_crosslinker

# 原子特征维度定义
N_FEATURES = 20

# 元素类型 one-hot (10维)
ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'other']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}

# 杂化方式 one-hot (4维)
HYBRIDIZATIONS = ['SP', 'SP2', 'SP3', 'other']
HYBRID_TO_IDX = {h: i for i, h in enumerate(HYBRIDIZATIONS)}


# =============================================================================
# 已处理序列记录管理
# =============================================================================

def load_processed_set(processed_file: Path) -> Set[str]:
    """加载已处理序列集合"""
    processed_file = Path(processed_file)
    if not processed_file.exists():
        return set()

    with open(processed_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def save_processed_set(processed_file: Path, seq_set: Set[str]):
    """保存已处理序列集合"""
    processed_file = Path(processed_file)
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_file, 'w') as f:
        for seq in sorted(seq_set):
            f.write(f"{seq}\n")


def append_processed(processed_file: Path, sequences: List[str]):
    """追加已处理序列到文件（不覆盖已有记录）"""
    processed_file = Path(processed_file)
    processed_file.parent.mkdir(parents=True, exist_ok=True)

    # 读取已有记录
    existing = set()
    if processed_file.exists():
        with open(processed_file, 'r') as f:
            existing = set(line.strip() for line in f if line.strip())

    # 追加新序列
    new_seqs = [seq for seq in sequences if seq not in existing]
    if new_seqs:
        with open(processed_file, 'a') as f:
            for seq in new_seqs:
                f.write(f"{seq}\n")


# =============================================================================
# 缓存管理（累积保存，不覆盖）
# =============================================================================

def load_cache(cache_file: Path) -> Tuple[List, List, List, Set]:
    """
    加载缓存数据

    Returns:
        (features_list, coords_list, energies, sequences_set)
    """
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return [], [], [], set()

    data = np.load(cache_file, allow_pickle=True)
    features_list = list(data['features']) if 'features' in data else []
    coords_list = list(data['coords']) if 'coords' in data else []
    energies = list(data['energies']) if 'energies' in data else []
    sequences = set(data['sequences']) if 'sequences' in data else set()

    print(f"  【缓存】加载 {len(energies)} 个已处理样本")
    return features_list, coords_list, energies, sequences


def save_cache(cache_file: Path, features_list: List, coords_list: List,
               energies: List, sequences: Set):
    """保存缓存数据（完全覆盖，保持一致性）"""
    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # 转换为列表以保证可序列化
    seq_list = list(sequences)

    np.savez_compressed(
        cache_file,
        features=np.array(features_list, dtype=object),
        coords=np.array(coords_list, dtype=object),
        energies=np.array(energies),
        sequences=np.array(seq_list, dtype=object)
    )


def append_to_cache(cache_file: Path, features: np.ndarray, coords: np.ndarray,
                    energy: float, sequence: str):
    """
    追加单个样本到缓存（原子操作，避免数据丢失）
    """
    cache_file = Path(cache_file)

    # 加载已有数据
    features_list, coords_list, energies, seq_set = load_cache(cache_file)

    # 检查是否已存在
    if sequence in seq_set:
        return False

    # 追加新数据
    features_list.append(features)
    coords_list.append(coords)
    energies.append(energy)
    seq_set.add(sequence)

    # 保存
    save_cache(cache_file, features_list, coords_list, energies, seq_set)
    return True


# =============================================================================
# 核心处理函数
# =============================================================================

def extract_atom_features(mol) -> Tuple[np.ndarray, np.ndarray]:
    """从RDKit分子中提取原子特征和坐标"""
    from rdkit import Chem

    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, N_FEATURES))
    coords = np.zeros((n_atoms, 3))

    if mol.GetNumConformers() == 0:
        raise ValueError("分子没有3D构象")

    conf = mol.GetConformer()

    for i, atom in enumerate(mol.GetAtoms()):
        element = atom.GetSymbol()
        elem_idx = ELEMENT_TO_IDX.get(element, ELEMENT_TO_IDX['other'])
        features[i, elem_idx] = 1.0

        hybrid = atom.GetHybridization()
        hybrid_str = str(hybrid)
        if 'SP3' in hybrid_str:
            hybrid_idx = 2
        elif 'SP2' in hybrid_str:
            hybrid_idx = 1
        elif 'SP' in hybrid_str:
            hybrid_idx = 0
        else:
            hybrid_idx = 3
        features[i, 10 + hybrid_idx] = 1.0

        features[i, 14] = atom.GetFormalCharge()

        is_hbd = False
        if atom.GetSymbol() in ['N', 'O']:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    is_hbd = True
                    break
        features[i, 15] = float(is_hbd)

        is_hba = atom.GetSymbol() in ['N', 'O']
        features[i, 16] = float(is_hba)

        features[i, 17] = float(atom.GetIsAromatic())
        features[i, 18] = atom.GetDegree() / 4.0
        features[i, 19] = atom.GetMass() / 100.0

        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]

    return features, coords


def process_sequence(sequence: str,
                     energy: float,
                     crosslinker: Optional[str] = None,
                     crosslinker_positions: Optional[List[int]] = None,
                     random_seed: int = 42) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """处理单个序列"""
    try:
        mol = build_peptide_with_rdkit(sequence)
        if crosslinker and crosslinker_positions:
            mol = add_crosslinker(mol, crosslinker, crosslinker_positions, sequence)

        mol = generate_3d_conformation(mol, random_seed)
        features, coords = extract_atom_features(mol)

        return features, coords, energy

    except Exception as e:
        print(f"  【失败】{e}")
        return None


# =============================================================================
# 主函数
# =============================================================================
def main(sequences_file: Optional[Path] = None,
         energies_file: Optional[Path] = None,
         output_dir: Optional[Path] = None,
         cache_file: Optional[Path] = None,
         processed_file: Optional[Path] = None,
         batch_size: int = 10,
         target_name: Optional[str] = None):
    """
    主函数 - 增量处理，不删除源文件

    Args:
        target_name: 靶点名称（用于确定EGNN数据目录）
    """
    # 如果未指定target_name，尝试从energies_file路径提取
    if target_name is None and energies_file is not None:
        # 尝试从路径提取: results/{target_name}/energies.csv
        parent = energies_file.parent
        if parent.parent.name == "results":
            target_name = parent.name
        else:
            target_name = "default"

    # 使用target_name确定EGNN目录
    if target_name is not None:
        egnn_dirs = config.get_egnn_dirs(target_name)
        if output_dir is None:
            output_dir = egnn_dirs["raw"]
        if cache_file is None:
            cache_file = output_dir / "processed_cache.npz"
        if processed_file is None:
            processed_file = output_dir / "processed_sequences.txt"
    else:
        # fallback：使用原默认路径
        if output_dir is None:
            output_dir = config.BASE_DIR / "egnn" / "raw"
        if cache_file is None:
            cache_file = output_dir / "processed_cache.npz"
        if processed_file is None:
            processed_file = output_dir / "processed_sequences.txt"

    # ... 其余代码保持不变
    print("EGNN数据准备 (增量处理版 - 不删除源文件)")
    print("=" * 60)

    # 1. 加载已处理序列
    print(f"\n[1/5] 加载已处理记录...")
    processed_set = load_processed_set(processed_file)
    print(f"  已处理: {len(processed_set)} 个序列")

    # 2. 加载源数据
    print(f"\n[2/5] 加载源数据...")
    data = load_data(sequences_file, energies_file)
    if not data:
        print("  没有有效数据")
        return

    sequences = [seq for seq, _ in data]
    energies = {seq: eng for seq, eng in data}
    print(f"  总序列: {len(sequences)} 个")

    # 3. 筛选未处理的序列
    pending = [(seq, energies[seq]) for seq in sequences if seq not in processed_set]
    print(f"  待处理: {len(pending)} 个")

    if not pending:
        print("\n  ✅ 所有序列已处理完成！")
        # 直接划分数据集
        split_and_save_dataset(cache_file, output_dir)
        return

    # 4. 处理序列（每 batch_size 个保存一次）
    print(f"\n[3/5] 处理序列 (每{batch_size}个保存一次)...")
    crosslinker = config.CROSSLINKER
    crosslinker_positions = config.CROSSLINKER_POSITIONS

    # 加载已有缓存
    features_list, coords_list, energies_list, seq_set = load_cache(cache_file)

    success_count = 0
    fail_count = 0
    total = len(pending)
    new_sequences = []  # 记录本次新处理的序列

    for i, (seq, energy) in enumerate(pending, 1):
        print(f"  处理 {i}/{total}: {seq[:20]}...", end=" ")

        # 检查是否已在缓存中（双重保险）
        if seq in seq_set:
            print("⏭️ 已缓存，跳过")
            continue

        result = process_sequence(seq, energy, crosslinker, crosslinker_positions)

        if result:
            features, coords, eng = result
            features_list.append(features)
            coords_list.append(coords)
            energies_list.append(eng)
            seq_set.add(seq)
            new_sequences.append(seq)
            success_count += 1
            print("✓")
        else:
            fail_count += 1
            print("✗")

        # 每 batch_size 个保存一次缓存和已处理记录
        if i % batch_size == 0 or i == total:
            save_cache(cache_file, features_list, coords_list, energies_list, seq_set)
            # 追加已处理记录
            if new_sequences:
                append_processed(processed_file, new_sequences)
                new_sequences = []  # 清空，避免重复追加
            print(f"    【保存】已处理 {success_count}/{total}，缓存已更新")

    # 最后再保存一次（确保所有数据都写入）
    save_cache(cache_file, features_list, coords_list, energies_list, seq_set)
    if new_sequences:
        append_processed(processed_file, new_sequences)

    print(f"\n  处理完成: 成功 {success_count}，失败 {fail_count}")
    print(f"  缓存总计: {len(seq_set)} 个样本")

    # 5. 划分并保存数据集
    print(f"\n[4/5] 划分并保存数据集...")
    split_and_save_dataset(cache_file, output_dir)

    # 6. 统计
    print(f"\n[5/5] 统计信息...")
    print(f"  已处理记录: {len(load_processed_set(processed_file))} 个")
    print(f"  缓存样本: {len(seq_set)} 个")

    print("\n" + "=" * 60)
    print("EGNN数据准备完成!")
    print(f"缓存文件: {cache_file}")
    print(f"已处理记录: {processed_file}")
    print("=" * 60)


def load_data(sequences_file: Path, energies_file: Path) -> List[Tuple[str, float]]:
    """加载序列和能量数据（不修改源文件）"""
    # 读取序列
    sequences = []
    with open(sequences_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and ',' not in line:
                sequences.append(line)

    # 读取能量
    energies = {}
    with open(energies_file, 'r') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                seq = parts[0]
                try:
                    energy = float(parts[1])
                    energies[seq] = energy
                except ValueError:
                    continue

    # 匹配
    data = []
    for seq in sequences:
        if seq in energies:
            data.append((seq, energies[seq]))

    return data


def split_and_save_dataset(cache_file: Path, output_dir: Path,
                           train_ratio: float = 0.8, val_ratio: float = 0.1):
    """从缓存中读取所有数据，按比例划分并保存"""
    features_list, coords_list, energies_list, seq_set = load_cache(cache_file)

    if len(energies_list) == 0:
        print("  错误: 缓存中没有数据")
        return

    total = len(energies_list)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    print(f"  总样本: {total}")
    print(f"  训练: {n_train}")
    print(f"  验证: {n_val}")
    print(f"  测试: {total - n_train - n_val}")

    # 打乱数据
    indices = list(range(total))
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_data = [(features_list[i], coords_list[i], energies_list[i]) for i in train_idx]
    val_data = [(features_list[i], coords_list[i], energies_list[i]) for i in val_idx]
    test_data = [(features_list[i], coords_list[i], energies_list[i]) for i in test_idx]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, name):
        if not data:
            print(f"    【警告】{name} 为空")
            return
        features_arr = np.array([d[0] for d in data], dtype=object)
        coords_arr = np.array([d[1] for d in data], dtype=object)
        energies_arr = np.array([d[2] for d in data])
        np.savez_compressed(
            output_dir / name,
            features=features_arr,
            coords=coords_arr,
            energies=energies_arr,
            allow_pickle=True
        )
        print(f"    ✓ {name}: {len(data)} 个样本")

    save_data(train_data, "train_data.npz")
    save_data(val_data, "val_data.npz")
    save_data(test_data, "test_data.npz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EGNN数据准备（增量处理版）')
    parser.add_argument('-s', '--sequences', type=Path, default=None,
                        help='序列文件路径')
    parser.add_argument('-e', '--energies', type=Path, default=None,
                        help='能量文件路径')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='输出目录')
    parser.add_argument('--cache', type=Path, default=None,
                        help='缓存文件路径')
    parser.add_argument('--processed', type=Path, default=None,
                        help='已处理序列记录文件路径')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='每批处理数量（默认10）')
    parser.add_argument('--target', type=str, default=None,
                        help='靶点名称（用于确定EGNN目录）')

    args = parser.parse_args()
    main(
        sequences_file=args.sequences,
        energies_file=args.energies,
        output_dir=args.output,
        cache_file=args.cache,
        processed_file=args.processed,
        batch_size=args.batch_size,
        target_name=args.target  # ← 新增
    )
