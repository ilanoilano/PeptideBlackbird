#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配体生成器 (ligand_generator.py)
功能：序列 → 3D构象（含交联剂）→ PDBQT
"""

import os
import sys
import subprocess
import tempfile
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import config

# 导入MCTS日志模块
try:
    from mcts_logger import get_logger, log_crosslinker_debug
except ImportError:
    get_logger = None
    log_crosslinker_debug = None


# 【修复】使用项目目录下的临时文件夹，避免硬编码 /tmp
TEMP_DIR = config.BASE_DIR / "temp" / "ligand_generator"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def is_carboxyl_carbon(mol, atom_idx):
    """
    检查一个碳原子是否是羧基碳（C(=O)O）

    Args:
        mol: RDKit分子
        atom_idx: 碳原子索引

    Returns:
        bool: 是否是羧基碳
    """
    from rdkit import Chem

    atom = mol.GetAtomWithIdx(atom_idx)

    if atom.GetAtomicNum() != 6:
        return False

    has_double_o = False
    has_single_o = False

    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() != 8:
            continue

        bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            has_single_o = True

    return has_double_o and has_single_o


def find_carboxyl_carbon(mol):
    """
    找到C端羧基碳（排除侧链羧基，如Asp/Glu）

    鲁棒版：只返回主链羧基碳

    Args:
        mol: RDKit分子

    Returns:
        (羧基碳索引, 单键氧索引)，找不到返回 (None, None)
    """
    from rdkit import Chem
    candidates = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue

        if not is_carboxyl_carbon(mol, atom.GetIdx()):
            continue

        # 找到单键氧（OH）
        o_single = None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 8:
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    o_single = neighbor.GetIdx()
                    break

        if o_single is None:
            continue

        # 检查这个羧基碳是否连接α碳
        # α碳的特征：连接一个非芳香氮
        is_main_chain = False
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 6:
                continue

            for alpha_neighbor in neighbor.GetNeighbors():
                if alpha_neighbor.GetAtomicNum() != 7:
                    continue
                if alpha_neighbor.GetIsAromatic():
                    continue
                is_main_chain = True
                break

            if is_main_chain:
                break

        if is_main_chain:
            candidates.append((atom.GetIdx(), o_single))

    if not candidates:
        # 回退到旧逻辑
        return _fallback_find_carboxyl_carbon(mol)

    return candidates[0]


def _fallback_find_carboxyl_carbon(mol):
    """回退逻辑：旧版 find_carboxyl_carbon"""
    from rdkit import Chem

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            o_double = None
            o_single = None
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 8:
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        o_double = neighbor.GetIdx()
                    elif bond.GetBondType() == Chem.BondType.SINGLE:
                        o_single = neighbor.GetIdx()
            if o_double is not None and o_single is not None:
                is_main_chain = False
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 6:
                        for ca_neighbor in neighbor.GetNeighbors():
                            if ca_neighbor.GetAtomicNum() == 7:
                                is_main_chain = True
                                break
                if is_main_chain:
                    return atom.GetIdx(), o_single
    return None, None


def find_amino_nitrogen(mol):
    """
    找到α-氨基氮（N端）

    鲁棒版：只返回真正的α-氨基氮，不返回侧链氨基

    Args:
        mol: RDKit分子

    Returns:
        α-氨基氮的原子索引，找不到返回None
    """
    candidates = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue

        # 跳过芳香氮
        if atom.GetIsAromatic():
            continue

        # 跳过带正电的氮
        if atom.GetFormalCharge() > 0:
            continue

        # 检查这个氮是否连接α碳
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 6:
                continue

            alpha_carbon = neighbor

            # 检查α碳是否连接羧基碳
            has_carboxyl = False
            for alpha_neighbor in alpha_carbon.GetNeighbors():
                if alpha_neighbor.GetAtomicNum() != 6:
                    continue

                if is_carboxyl_carbon(mol, alpha_neighbor.GetIdx()):
                    has_carboxyl = True
                    break

            if has_carboxyl:
                candidates.append((atom.GetIdx(), alpha_carbon.GetIdx()))
                break

    if not candidates:
        return _fallback_find_amino_nitrogen(mol)

    candidates.sort(key=lambda x: x[0])
    return candidates[0][0]


def _fallback_find_amino_nitrogen(mol):
    """回退逻辑：旧版 find_amino_nitrogen"""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:
            neighbors = list(atom.GetNeighbors())
            carbon_count = sum(1 for n in neighbors if n.GetAtomicNum() == 6)
            if carbon_count >= 1 and carbon_count <= 2:
                return atom.GetIdx()
    return None


# 氨基酸SMILES（N端游离，C端羧基）
# 天然氨基酸SMILES（硬编码，带占位符 [N]）
AA_SMILES_NATURAL = {
    'A': '[N][C@@H](C)C(=O)O',
    'C': '[N][C@@H](CS)C(=O)O',
    'D': '[N][C@@H](CC(=O)O)C(=O)O',
    'E': '[N][C@@H](CCC(=O)O)C(=O)O',
    'F': '[N][C@@H](Cc1ccccc1)C(=O)O',
    'G': '[N]CC(=O)O',
    'H': '[N][C@@H](Cc1c[nH]cn1)C(=O)O',
    'I': '[N][C@@H](C(C)CC)C(=O)O',
    'K': '[N][C@@H](CCCCN)C(=O)O',
    'L': '[N][C@@H](CC(C)C)C(=O)O',
    'M': '[N][C@@H](CCSC)C(=O)O',
    'N': '[N][C@@H](CC(=O)N)C(=O)O',
    'P': 'N1CCCC1C(=O)O',
    'Q': '[N][C@@H](CCC(=O)N)C(=O)O',
    'R': '[N][C@@H](CCCNC(=N)N)C(=O)O',
    'S': '[N][C@@H](CO)C(=O)O',
    'T': '[N][C@@H](C(C)O)C(=O)O',
    'V': '[N][C@@H](C(C)C)C(=O)O',
    'W': '[N][C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': '[N][C@@H](Cc1ccc(O)cc1)C(=O)O',
}

# 非天然氨基酸SMILES（运行时从 amino/AMINO.txt 加载）
# key 是氨基酸名称（如 "Aib", "Nle"）
AA_SMILES_NONNATURAL = {}
# 合并后的字典（运行时由 load_nonnatural_smiles 填充）
AA_SMILES = {**AA_SMILES_NATURAL, **AA_SMILES_NONNATURAL}


# =============================================================================
# 交联剂 SMILES
# =============================================================================
# 【说明】TBMB 使用 Kekulé 形式（明确指定双键），避免芳香环 kekulization 问题
CROSSLINKER_SMILES = {
    "TBMB": "BrCC1=CC(CBr)=CC(CBr)=C1",       # Kekulé 形式
    "TATA": "C(CS)(CS)CS",
    "TBAB": "C1=C(CBr)C=C(CBr)C(CBr)=C1CBr",   # Kekulé 形式
}








def _convert_to_placeholder_smiles(smiles: str) -> str:
    """
    把完整氨基酸 SMILES 转成带 [N] 占位符的形式

    输入: "CC(C)(N)C(=O)O"
    输出: "CC(C)([N])C(=O)O"（或类似）

    关键：
    - RDKit 输出 N 时通常带显式氢（如 [NH2:4]）
    - 需要把 [N(Hn):map] 统一替换为 [N]
    """
    from rdkit import Chem
    import re

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"无法解析 SMILES: {smiles}")

    n_idx = find_amino_nitrogen(mol)
    if n_idx is None:
        raise RuntimeError(f"找不到 α-氨基氮: {smiles}")

    # 给所有原子加原子映射号（1-based）
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    mapped_smiles = Chem.MolToSmiles(mol)
    map_num = n_idx + 1

    # 【修正】匹配 [N...:map_num]，其中 ... 可能是空、H、H2、H3
    pattern = rf'\[N[H0-9]*:{map_num}\]'

    if not re.search(pattern, mapped_smiles):
        raise RuntimeError(
            f"无法定位 α-氨基 N 的映射标记。"
            f"mapped_smiles={mapped_smiles}, n_idx={n_idx}, map_num={map_num}"
        )

    new_smiles = re.sub(pattern, '[N]', mapped_smiles)

    # 去掉其他原子映射号
    new_smiles = re.sub(r':\d+\]', ']', new_smiles)
    new_smiles = re.sub(r':\d+', '', new_smiles)

    return new_smiles


def load_nonnatural_smiles(txt_path=None) -> dict:
    """
    从 amino/AMINO.txt 加载非天然氨基酸的SMILES，并转成占位符形式
    """
    from amino_acid_loader import parse_amino_txt

    amino_dict = parse_amino_txt(txt_path)

    AA_SMILES_NONNATURAL.clear()
    for name, smiles in amino_dict.items():
        try:
            placeholder_smiles = _convert_to_placeholder_smiles(smiles)
            AA_SMILES_NONNATURAL[name] = placeholder_smiles
            print(f"  [{name}] {smiles} -> {placeholder_smiles}")
        except Exception as e:
            print(f"  [警告] {name} 转换失败: {e}，使用原 SMILES")
            AA_SMILES_NONNATURAL[name] = smiles

    AA_SMILES.clear()
    AA_SMILES.update(AA_SMILES_NATURAL)
    AA_SMILES.update(AA_SMILES_NONNATURAL)

    print(f"[ligand_generator] 已加载 {len(AA_SMILES_NONNATURAL)} 个非天然氨基酸SMILES")
    return AA_SMILES_NONNATURAL



def build_peptide_with_rdkit(sequence) -> 'Chem.Mol':
    """
    从氨基酸序列构建肽分子

    支持：
    - 天然氨基酸（单大写字母，如 "A", "C", "D"）
    - 非天然氨基酸（大写字母 + 后续小写字母，如 "Aib", "Nle"）

    Args:
        sequence: 氨基酸序列
                  - 字符串形式：如 "ACAibCG"
                  - 列表形式：如 ["A", "C", "Aib", "C", "G"]

    Returns:
        RDKit 分子对象

    Raises:
        RuntimeError: RDKit 未安装
        ValueError: 序列为空
        RuntimeError: 未知氨基酸或 SMILES 解析失败
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("RDKit未安装")

    # 【关键改动1】解析为氨基酸名称列表
    amino_acids = config.parse_sequence(sequence)

    if not amino_acids:
        raise ValueError("序列为空")

    try:
        # ============================================================
        # 情况1：单个氨基酸
        # ============================================================
        if len(amino_acids) == 1:
            aa = amino_acids[0]
            smiles = AA_SMILES.get(aa)
            if smiles is None:
                raise RuntimeError(f"未知氨基酸: '{aa}'（不在 AA_SMILES 中）")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError(f"无法解析氨基酸 SMILES: {aa} -> {smiles}")
            return mol

        # ============================================================
        # 情况2：多个氨基酸，逐个拼接
        # ============================================================
        # 第一个氨基酸
        first_aa = amino_acids[0]
        first_smiles = AA_SMILES.get(first_aa)
        if first_smiles is None:
            raise RuntimeError(f"未知氨基酸: '{first_aa}'（位置 0）")
        mol = Chem.MolFromSmiles(first_smiles)
        if mol is None:
            raise RuntimeError(f"无法解析第一个氨基酸: {first_aa} -> {first_smiles}")

        # 逐个添加剩余氨基酸
        for i in range(1, len(amino_acids)):
            aa = amino_acids[i]
            aa_smiles = AA_SMILES.get(aa)
            if aa_smiles is None:
                raise RuntimeError(f"未知氨基酸: '{aa}'（位置 {i}）")

            next_mol = Chem.MolFromSmiles(aa_smiles)
            if next_mol is None:
                raise RuntimeError(f"无法解析氨基酸: {aa}（位置 {i}） -> {aa_smiles}")

            n_prev = mol.GetNumAtoms()
            c_atom_idx, oh_atom_idx = find_carboxyl_carbon(mol)
            n_atom_idx = find_amino_nitrogen(next_mol)
            if c_atom_idx is None or oh_atom_idx is None:
                raise RuntimeError(
                    f"位置 {i}（氨基酸 '{aa}'）：找不到当前肽链的 C 端羧基"
                )
            if n_atom_idx is None:
                raise RuntimeError(
                    f"位置 {i}（氨基酸 '{aa}'）：找不到 α-氨基氮"
                )

            # 【关键】把 next_mol 的 N 索引映射到 combined 坐标系
            n_idx_in_combined = n_atom_idx + n_prev

            # 合并两个分子
            combined = Chem.CombineMols(mol, next_mol)
            editable = Chem.EditableMol(combined)

            # 删除羟基（OH）
            editable.RemoveAtom(oh_atom_idx)

            # 【关键】删除 OH 后，所有索引 > oh_atom_idx 的原子索引减 1
            c_atom_idx_adj = c_atom_idx
            n_idx_adj = n_idx_in_combined
            if c_atom_idx_adj > oh_atom_idx:
                c_atom_idx_adj -= 1
            if n_idx_adj > oh_atom_idx:
                n_idx_adj -= 1

            # 创建肽键（C-N）
            editable.AddBond(c_atom_idx_adj, n_idx_adj, Chem.BondType.SINGLE)
            mol = editable.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            # 尝试跳过 kekulization
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                               Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception as e2:
                raise RuntimeError(f"分子 sanitize 失败: {e2}")
        return mol
    except RuntimeError:
        # 直接重新抛出我们的自定义错误
        raise
    except Exception as e:
        # 其他异常统一包装
        raise RuntimeError(f"构建肽链失败: {e}")

def find_cys_sulfur_atoms(mol, sequence):
    """
    找到所有 Cys 的硫原子索引（按序列中 Cys 的出现顺序）

    关键：
    - 只找 Cys 的 S，排除 Met 的 S
    - Cys 的 S 连接在 CB 上（CB 连接 CA，CA 连接 N）
    - Met 的 S 连接在 CG 上（CG 连接 CB，CB 连接 CA，多一层）

    Args:
        mol: 肽分子
        sequence: 氨基酸序列

    Returns:
        List[硫原子索引]，按序列中 Cys 的出现顺序排列
    """
    from rdkit import Chem
    amino_acids = config.parse_sequence(sequence)
    # 方法：利用 RDKit 构建顺序，S 原子的出现顺序与序列中 Cys 的顺序一致
    # 但需要区分 Met 的 S（Met 的 S 在侧链末端，Cys 的 S 靠近主链）

    cys_sulfur_indices = []
    met_sulfur_indices = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 16:  # 只看 S
            continue

        # 获取 S 的邻居碳
        s_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
        if not s_neighbors:
            continue

        c_connected_to_s = s_neighbors[0]

        # 判断这个碳距离主链氮有多远
        # Cys: S-CB-CA-N (距离2)
        # Met: S-CG-CB-CA-N (距离3)

        # 广度优先搜索，找最近的主链 N
        from collections import deque
        visited = {atom.GetIdx()}
        queue = deque([(c_connected_to_s, 1)])  # (原子, 距离S的键数)

        min_dist_to_n = None
        while queue:
            current_atom, dist = queue.popleft()

            if current_atom.GetAtomicNum() == 7:  # 找到 N
                min_dist_to_n = dist
                break

            if dist > 4:  # 限制搜索深度
                continue

            for neighbor in current_atom.GetNeighbors():
                if neighbor.GetIdx() not in visited:
                    visited.add(neighbor.GetIdx())
                    queue.append((neighbor, dist + 1))

        # 距离 N 为 2 → Cys
        # 距离 N 为 3 → Met
        if min_dist_to_n == 3:
            cys_sulfur_indices.append(atom.GetIdx())
        elif min_dist_to_n == 4:
            met_sulfur_indices.append(atom.GetIdx())
        else:
            # 无法判断，默认当作 Cys
            cys_sulfur_indices.append(atom.GetIdx())

    # 验证
    expected_cys = sum(1 for aa in amino_acids if aa == 'C')
    if len(cys_sulfur_indices) != expected_cys:
        print(f"【警告】序列中有 {expected_cys} 个 Cys，"
              f"但找到 {len(cys_sulfur_indices)} 个 Cys 的 S，"
              f"{len(met_sulfur_indices)} 个 Met 的 S")

    return cys_sulfur_indices


def add_crosslinker(mol: 'Chem.Mol',
                    crosslinker_type: str,
                    cys_positions: List[int],
                    sequence: str) -> 'Chem.Mol':
    from rdkit import Chem
    amino_acids = config.parse_sequence(sequence)
    if crosslinker_type not in CROSSLINKER_SMILES:
        print(f"【警告】未知交联剂类型: {crosslinker_type}，跳过添加")
        return mol
    xlinker_smiles = CROSSLINKER_SMILES[crosslinker_type]
    xlinker_mol = Chem.MolFromSmiles(xlinker_smiles)
    if xlinker_mol is None:
        print(f"【警告】无法解析交联剂SMILES: {xlinker_smiles}")
        return mol

    # 肽链内部，获取S原子【原始肽mol的局部ID】
    cys_sulfur_local_indices = find_cys_sulfur_atoms(mol, sequence)
    if len(cys_sulfur_local_indices) < 3:
        print(f"【警告】只有 {len(cys_sulfur_local_indices)} 个 Cys，需要 3 个")
        return mol

    selected_sulfur_local = []
    for pos in cys_positions:
        if pos < len(amino_acids) and amino_acids[pos] == 'C':
            cys_count = 0
            for i, aa in enumerate(amino_acids):
                if aa == 'C':
                    if i == pos and cys_count < len(cys_sulfur_local_indices):
                        selected_sulfur_local.append(cys_sulfur_local_indices[cys_count])
                        break
                    cys_count +=1

    print(f"【DEBUG‑交联】肽内部S局部索引 selected_sulfur_local={selected_sulfur_local}")
    if len(selected_sulfur_local) <3:
        print(f"【警告】只找到 {len(selected_sulfur_local)} 个Cys硫")
        return mol

    n_peptide_atoms = mol.GetNumAtoms()
    # CombineMols(A,B): A全部复制到前面，ID完全不变；B从 n_peptide_atoms 开始
    combined = Chem.CombineMols(mol, xlinker_mol)
    rw_mol = Chem.RWMol(combined)

    # ✅肽S在combined中ID = 原始局部ID！！不加任何偏移！
    sulfur_global = selected_sulfur_local[:3]
    print(f"【DEBUG】合并后肽S全局ID（无偏移） sulfur_global={sulfur_global}")

    br_indices = []
    c_indices_xlinker_local = []
    # 只遍历TBMB部分：[n_peptide_atoms ... end]
    for idx in range(n_peptide_atoms, rw_mol.GetNumAtoms()):
        atom = rw_mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 35:
            br_indices.append(idx)
            for nb in atom.GetNeighbors():
                c_indices_xlinker_local.append(nb.GetIdx())

    print(f"【DEBUG】Br全局索引 br_indices={br_indices}")
    print(f"【DEBUG】TBMB与Br相连C原始全局ID c_indices_xlinker_local={c_indices_xlinker_local}")

    # 逆序删除Br（大ID优先删，降低扰动）
    for br_idx in sorted(br_indices, reverse=True):
        rw_mol.RemoveAtom(br_idx)

    # 只修正TBMB内部C的ID：统计在该C前面被删掉多少Br
    adjusted_c_global = []
    for c_orig in c_indices_xlinker_local:
        del_cnt = sum(1 for b in br_indices if b < c_orig)
        adjusted_c_global.append(c_orig - del_cnt)
    print(f"【DEBUG】TBMB修正后C全局ID adjusted_c_global={adjusted_c_global}")

    num_atoms_now = rw_mol.GetNumAtoms()
    # 边界检查
    for s_g, c_g in zip(sulfur_global, adjusted_c_global):
        if s_g >= num_atoms_now or c_g >= num_atoms_now:
            raise RuntimeError(f"原子越界! S={s_g}, C={c_g}, total={num_atoms_now}")
        print(f"【DEBUG‑FINAL BOND】连接 S@{s_g} <--> C@{c_g}")
        rw_mol.AddBond(s_g, c_g, Chem.BondType.SINGLE)

    result_mol = rw_mol.GetMol()
    frags = Chem.GetMolFrags(result_mol, asMols=False, sanitizeFrags=False)
    print(f"【DEBUG】交联后分子碎片数量: {len(frags)}")
    if len(frags) > 1:
        print(f"【ERROR】碎片数量={len(frags)}，拓扑出错！")
        for idx,frag in enumerate(frags):
            print(f"  碎片{idx+1}: {frag}")
        raise RuntimeError(f"交联产生多个碎片，碎片数={len(frags)}")

    # sanitize
    try:
        Chem.SanitizeMol(result_mol)
    except Exception as e:
        print(f"【警告】Sanitize失败: {e}")
        try:
            Chem.SanitizeMol(
                result_mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
        except:
            pass
    return result_mol

def _validate_conformation_quality(mol: 'Chem.Mol', min_z_range: float = 2.0) -> bool:
    """
    验证3D构象质量，确保不是2D构象
    
    Args:
        mol: 带3D构象的分子
        min_z_range: Z轴最小范围（Å）
    
    Returns:
        True if 构象质量合格
    """
    from rdkit import Chem
    import numpy as np
    
    if mol.GetNumConformers() == 0:
        return False
    
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    
    coords_array = np.array(coords)
    x_range = coords_array[:, 0].max() - coords_array[:, 0].min()
    y_range = coords_array[:, 1].max() - coords_array[:, 1].min()
    z_range = coords_array[:, 2].max() - coords_array[:, 2].min()
    
    print(f"【ligand_generator】构象坐标范围 - X: {x_range:.2f}Å, Y: {y_range:.2f}Å, Z: {z_range:.2f}Å")
    
    # 检查Z轴范围（2D构象的Z范围接近0）
    if z_range < min_z_range:
        print(f"【警告】Z轴范围过小 ({z_range:.2f}Å < {min_z_range}Å)，可能是2D构象")
        return False
    
    # 检查各轴范围是否合理（避免极端扁平构象）
    if x_range < 1.0 or y_range < 1.0:
        print(f"【警告】X或Y轴范围异常 (X={x_range:.2f}Å, Y={y_range:.2f}Å)")
        return False
    
    return True


def generate_3d_conformation(mol: 'Chem.Mol', random_seed: int = 42, target_name: str = None) -> 'Chem.Mol':
    """
    生成3D构象
    
    策略:
    1. 首先尝试 RDKit ETKDGv3
    2. 如果失败，回退到 OpenBabel --gen3D
    
    Returns:
        带3D构象的分子
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import tempfile

    
    # 【修复】更健壮的sanitization处理
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"【警告】构象生成前sanitization失败: {e}")
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                           Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
            print(f"【警告】跳过kekulization后继续")
        except Exception as e2:
            print(f"【警告】跳过kekulization后仍然失败: {e2}")
            print(f"【警告】将继续尝试生成构象，但可能失败")
    
    # 添加氢原子
    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f"【错误】添加氢原子失败: {e}")
        raise RuntimeError(f"无法为分子添加氢原子: {e}")
    
    # ===== 方法1: RDKit ETKDGv3 =====
    success = False
    
    try:
        from rdkit.Chem import rdDistGeom
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = random_seed
        params.enforceChirality = False
        result = rdDistGeom.EmbedMolecule(mol, params)
        if result == 0:
            # 【关键修复】验证构象质量，确保不是2D构象
            if _validate_conformation_quality(mol):
                success = True
                print(f"【ligand_generator】✓ RDKit ETKDGv3 构象生成成功且质量合格")
            else:
                print(f"【警告】RDKit ETKDGv3 返回成功但构象质量不合格，视为失败")
    except Exception as e:
        print(f"【警告】RDKit ETKDGv3 失败: {e}")
    
    # ===== 方法2: RDKit 标准 Embed =====
    if not success:
        try:
            result = AllChem.EmbedMolecule(mol, randomSeed=random_seed, maxAttempts=100)
            if result == 0:
                # 【关键修复】验证构象质量
                if _validate_conformation_quality(mol):
                    success = True
                    print(f"【ligand_generator】✓ RDKit 标准 Embed 构象生成成功且质量合格")
                else:
                    print(f"【警告】RDKit 标准 Embed 返回成功但构象质量不合格，视为失败")
        except Exception as e:
            print(f"【警告】RDKit 标准 Embed 失败: {e}")
    
    # ===== 方法3: RDKit 随机坐标 =====
    if not success:
        try:
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100, randomSeed=random_seed)
            if result == 0:
                # 【关键修复】验证构象质量
                if _validate_conformation_quality(mol):
                    success = True
                    print(f"【ligand_generator】✓ RDKit 随机坐标构象生成成功且质量合格")
                else:
                    print(f"【警告】RDKit 随机坐标返回成功但构象质量不合格，视为失败")
        except Exception as e:
            print(f"【警告】RDKit 随机坐标 Embed 失败: {e}")
    
    # ===== 方法4: OpenBabel 回退 =====
    if not success:
        print(f"【ligand_generator】RDKit 所有方法失败，回退到 OpenBabel...")
        
        try:
            # 导出为 SMILES
            smiles = Chem.MolToSmiles(mol)
            print(f"【ligand_generator】SMILES: {smiles[:80]}...")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                input_smi = os.path.join(tmpdir, "input.smi")
                output_sdf = os.path.join(tmpdir, "output.sdf")
                
                # 写入 SMILES
                with open(input_smi, 'w') as f:
                    f.write(smiles)
                
                # OpenBabel 生成3D构象
                cmd = [
                    "obabel",
                    "-ismi", input_smi,
                    "-osdf", "-O", output_sdf,
                    "--gen3D", "best",
                    "-h",
                    "--minimize",
                    "--ff", "MMFF94"
                ]
                
                print(f"【ligand_generator】运行 OpenBabel: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode != 0:
                        # 尝试快速模式
                        print(f"【ligand_generator】尝试 OpenBabel 快速模式...")
                        cmd_fast = [
                            "obabel",
                            "-ismi", input_smi,
                            "-osdf", "-O", output_sdf,
                            "--gen3D", "fast",
                            "-h"
                        ]
                        result = subprocess.run(
                            cmd_fast,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        
                        if result.returncode != 0:
                            raise RuntimeError(f"OpenBabel 失败: {result.stderr}")
                    
                    print(f"【ligand_generator】✓ OpenBabel 3D构象生成完成")
                    
                    # 读取3D构象回 RDKit
                    supplier = Chem.SDMolSupplier(output_sdf, removeHs=False)
                    mol_3d = next(supplier)
                    
                    if mol_3d is None:
                        raise RuntimeError("无法从 SDF 读取分子")
                    
                    if mol_3d.GetNumConformers() == 0:
                        raise RuntimeError("OpenBabel 生成的分子没有3D构象")
                    
                    mol = mol_3d
                    success = True
                    print(f"【ligand_generator】✓ OpenBabel 构象读取成功")
                    
                except FileNotFoundError:
                    print(f"【错误】OpenBabel 未安装，跳过")
                    raise RuntimeError("所有构象生成方法都失败，且 OpenBabel 未安装")
                    
        except Exception as e:
            print(f"【错误】OpenBabel 回退失败: {e}")
    
    if not success:
        raise RuntimeError("所有构象生成方法都失败（RDKit 和 OpenBabel）")
    
    # ===== 构象优化 =====
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        print(f"【ligand_generator】✓ MMFF 优化完成")
    except:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            print(f"【ligand_generator】✓ UFF 优化完成")
        except:
            print("【警告】构象优化失败，使用未优化的构象")
    
    # ===== 验证3D构象质量 =====
    try:
        import numpy as np
        conf = mol.GetConformer()
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        
        coords_array = np.array(coords)
        x_range = coords_array[:, 0].max() - coords_array[:, 0].min()
        y_range = coords_array[:, 1].max() - coords_array[:, 1].min()
        z_range = coords_array[:, 2].max() - coords_array[:, 2].min()
        
        print(f"【ligand_generator】坐标范围 - X: {x_range:.2f}Å, Y: {y_range:.2f}Å, Z: {z_range:.2f}Å")
        
        if z_range < 2.0:
            print(f"【警告】Z轴范围过小 ({z_range:.2f}Å)，3D构象可能异常")
        else:
            pass
            
    except Exception as e:
        print(f"【警告】无法验证3D构象质量: {e}")
    
    # ===== 计算 Gasteiger 电荷 =====
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        print(f"【警告】Gasteiger电荷计算失败: {e}")
        print(f"【警告】将继续生成PDBQT，但电荷可能为0")
    
    # ===== 平移到口袋中心（作为Vina初始位置）=====
    if target_name is not None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from vina import get_pocket_center
            
            pocket_center = get_pocket_center(target_name)
            if pocket_center is not None:
                # 计算当前质心
                conf = mol.GetConformer()
                coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                
                import numpy as np
                coords_array = np.array(coords)
                current_centroid = np.mean(coords_array, axis=0)
                
                # 计算平移向量（将质心移到口袋中心）
                translation = pocket_center - current_centroid
                print(f"  平移向量: ({translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f})")
                
                # 应用平移
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    new_pos = Chem.rdGeometry.Point3D(
                        pos.x + translation[0],
                        pos.y + translation[1],
                        pos.z + translation[2]
                    )
                    conf.SetAtomPosition(i, new_pos)
                
                # 验证新质心
                new_coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    new_coords.append([pos.x, pos.y, pos.z])
                new_centroid = np.mean(np.array(new_coords), axis=0)
            else:
                print(f"【警告】无法获取口袋中心，跳过平移")
        except Exception as e:
            print(f"【警告】平移到口袋中心失败: {e}")
    
    return mol


def rdkit_mol_to_pdbqt(mol: 'Chem.Mol', output_path: Path) -> Path:
    """
    【关键修复】直接使用RDKit生成PDBQT格式，保留Gasteiger电荷
    不经过OpenBabel，避免电荷丢失
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # PDBQT原子类型映射
    atom_type_map = {
        1: 'HD',  # 氢（给体）
        6: 'C',   # 碳
        7: 'N',   # 氮
        8: 'OA',  # 氧（受体）
        16: 'S',  # 硫
        17: 'Cl', # 氯
        35: 'Br', # 溴
        53: 'I',  # 碘
    }
    
    # 获取分子中的原子
    atoms = mol.GetAtoms()
    conf = mol.GetConformer()
    
    pdbqt_lines = []
    pdbqt_lines.append("REMARK  Generated by RDKit with Gasteiger charges")
    pdbqt_lines.append("REMARK  " + "-" * 50)
    pdbqt_lines.append("ROOT")
    
    # 写入原子
    # 写入原子（跳过氢原子，Vina使用非极性氢）
    atom_idx = 0
    for atom in atoms:
        # 跳过氢原子（原子序数=1）
        if atom.GetAtomicNum() == 1:
            continue
        
        atom_idx += 1
        pos = conf.GetAtomPosition(atom.GetIdx())
        
        # 获取原子类型
        atomic_num = atom.GetAtomicNum()
        atom_type = atom_type_map.get(atomic_num, 'A')
        
        # 获取Gasteiger电荷
        try:
            charge = atom.GetDoubleProp('_GasteigerCharge')
        except:
            charge = 0.0
        
        # 获取原子名称
        atom_name = atom.GetSymbol()
        
        # PDBQT格式: ATOM 序号 名称 残基 链 残基序号 x y z 占据 温度因子 电荷 类型
        line = f"ATOM  {atom_idx:5d}  {atom_name:3s} UNK A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:+.3f} {atom_type:2s}"
        pdbqt_lines.append(line)
    
    pdbqt_lines.append("ENDROOT")
    pdbqt_lines.append("TORSDOF 0")
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write('\n'.join(pdbqt_lines))
    
    return output_path


def mol_to_pdbqt(mol: 'Chem.Mol', output_path: Path) -> Path:
    """
    将分子转换为PDBQT
    
    【修改点3】优先使用RDKit直接生成，保留电荷信息
    如果失败，回退到OpenBabel方法
    """
    from rdkit import Chem
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 【关键修复】首先尝试使用RDKit直接生成PDBQT（保留电荷）
    try:
        return rdkit_mol_to_pdbqt(mol, output_path)
    except Exception as e:
        print(f"【警告】RDKit直接生成失败: {e}")
        print(f"【警告】回退到OpenBabel方法...")
    
    # 回退到OpenBabel方法
    temp_sdf = output_path.with_suffix('.temp.sdf')
    
    # 写入SDF（保留完整的分子结构信息，包括键序）
    writer = Chem.SDWriter(str(temp_sdf))
    writer.write(mol)
    writer.close()
    
    obabel_path = config.TOOLS.get("obabel", "obabel")
    
    # 【修改点2】删除obabel的-p参数，因为电荷已在RDKit中计算
    # 【修改点3】从SDF读取而不是PDB，保留键序信息
    cmd = [
        obabel_path,
        str(temp_sdf),
        "-opdbqt",
        # "-p",  # 删除：不再依赖OpenBabel计算电荷
        "-xl",
        "-O", str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            # 【修改点3】如果SDF转换失败，尝试PDB作为fallback
            print(f"【警告】SDF转换失败，尝试PDB格式: {result.stderr}")
            temp_pdb = output_path.with_suffix('.temp.pdb')
            Chem.MolToPDBFile(mol, str(temp_pdb))
            
            cmd_pdb = [
                obabel_path,
                str(temp_pdb),
                "-opdbqt",
                "-xl",
                "-O", str(output_path)
            ]
            result_pdb = subprocess.run(cmd_pdb, capture_output=True, text=True, timeout=60)
            
            if temp_pdb.exists():
                temp_pdb.unlink()
            
            if result_pdb.returncode != 0:
                raise RuntimeError(f"OpenBabel转换失败(SDF和PDB都失败): {result_pdb.stderr}")
        
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("PDBQT文件生成失败")
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        root_count = content.count('\nROOT\n')
        if root_count == 0:
            print(f"【警告】配体PDBQT缺少ROOT标签")
        elif root_count > 1:
            print(f"【错误】配体PDBQT包含多个ROOT标签（{root_count}个）")
            raise RuntimeError(f"配体PDBQT格式错误：包含{root_count}个ROOT标签")
        
    finally:
        if temp_sdf.exists():
            temp_sdf.unlink()
    
    return output_path


def generate_ligand(sequence: str, target_name: Optional[str] = None,
                    crosslinker: Optional[str] = None,
                    crosslinker_positions: Optional[List[int]] = None,
                    output_dir: Optional[Path] = None,
                    random_seed: int = 42) -> Path:
    """主函数：序列 → PDBQT"""
    import hashlib
    from rdkit import Chem
    
    if output_dir is None:
        output_dir = TEMP_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.md5(f"{config.format_sequence(sequence)}_{crosslinker}".encode()).hexdigest()[:8]
    output_pdbqt = output_dir / f"peptide_{seq_hash}.pdbqt"
    print(f"构建肽链：{sequence}")
    # 1. 构建肽链
    mol = build_peptide_with_rdkit(sequence)
    
    # 验证肽链
    smiles_check = Chem.MolToSmiles(mol)
    if '.' in smiles_check:
        print(f"【错误】肽链构建失败，存在未连接片段: {smiles_check}")
        raise RuntimeError("肽链构建失败")
    
    # 2. 添加交联剂
    if crosslinker and crosslinker in CROSSLINKER_SMILES:
        print(f"【ligand_generator】添加交联剂: {crosslinker}")
        
        if crosslinker_positions:
            mol = add_crosslinker(mol, crosslinker, crosslinker_positions, sequence)
            
            # 验证交联后的分子
            smiles_check = Chem.MolToSmiles(mol)
            if '.' in smiles_check:
                print(f"【错误】交联剂添加失败，存在未连接片段: {smiles_check}")
                raise RuntimeError("交联剂添加失败")
    
    # 3. 生成3D构象
    print(f"【ligand_generator】生成3D构象...")
    # 尝试获取 target_name
    target_name = getattr(config, "TARGET_NAME", None)
    if target_name is None:
            # 尝试从环境变量获取
        import os
        target_name = os.environ.get("TARGET_NAME", None)
    if target_name is None:
            # 尝试从 results 目录推断
        results_dir = getattr(config, "RESULTS_DIR", None)
        if results_dir and results_dir.exists():
            subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                target_name = subdirs[0].name
                print(f"【ligand_generator】推断 target_name: {target_name}")
        mol = generate_3d_conformation(mol, random_seed, target_name=target_name)
    if mol.GetNumConformers() == 0:
        raise RuntimeError("分子没有 3D 构象")

    # 检查坐标
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        if any(math.isnan(c) for c in [pos.x, pos.y, pos.z]):
            raise RuntimeError(f"原子 {i} 坐标包含 NaN")
    # 4. 转换为PDBQT
    pdbqt_path = mol_to_pdbqt(mol, output_pdbqt)
    
    print(f"【ligand_generator】✓ 完成: {pdbqt_path}")
    
    return pdbqt_path


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='配体生成器')
    parser.add_argument('-s', '--sequence', type=str, required=True,
                       help='氨基酸序列')
    parser.add_argument('-c', '--crosslinker', type=str, 
                       default=config.CROSSLINKER,
                       help=f'交联剂类型（默认: {config.CROSSLINKER}）')
    parser.add_argument('-o', '--output', type=Path, default=None,
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    try:
        pdbqt_path = generate_ligand(
            sequence=args.sequence,
            crosslinker=args.crosslinker,
            output_dir=args.output,
            random_seed=args.seed
        )
        print(f"✓ PDBQT生成成功: {pdbqt_path}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
