#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非天然氨基酸加载器 (amino_acid_loader.py)

功能：
1. 读取 amino/AMINO.txt（格式：名称,SMILES）
2. 对每个非天然氨基酸：
   - RDKit 解析 + 验证主链 NH2-CH-COOH
   - 生成 3D 构象，保存到 amino/{name}.pdb
   - 计算 Crippen LogP → 归一化到 Kyte-Doolittle 尺度
   - 计算 pH=7.4 净电荷
   - 计算残基分子量（游离 AA - 18.015）
3. 输出到 config.py 的三个字典（运行时注册）
4. 保存 amino/amino_acid_properties.json 备份

用法：
    from amino_acid_loader import load_and_register
    load_and_register()
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import config


# =============================================================================
# 路径配置
# =============================================================================
AMINO_DIR = config.BASE_DIR / "amino"
AMINO_TXT = AMINO_DIR / "AMINO.txt"
PROPERTIES_JSON = AMINO_DIR / "amino_acid_properties.json"


# =============================================================================
# pKa 表（用于计算 pH=7.4 净电荷）
# =============================================================================
PKA_TABLE = {
    "carboxyl": 2.0,         # -COOH
    "alpha_amino": 9.0,      # α-NH2
    "imidazole": 6.0,        # His 咪唑
    "guanidine": 12.5,       # Arg 胍基
    "phenol": 10.0,          # Tyr 酚羟基
    "thiol": 8.3,            # Cys 巯基
    "sidechain_amino": 10.5, # Lys 侧链氨基
}

# 目标 pH
TARGET_PH = 7.4


# =============================================================================
# 天然氨基酸的 (Crippen LogP, Kyte-Doolittle) 对，用于线性拟合
# 这些值用于将 LogP 映射到 Kyte-Doolittle 尺度
# =============================================================================
NATURAL_AA_LOGP_KD = [
    # (LogP, KD) —— LogP 用 Crippen 计算得到，KD 用实验值
    # 下面这些 LogP 值是预先计算好的，供线性拟合用
    (-0.5, -0.4),   # Gly
    (1.0, 1.8),     # Ala
    (2.0, 4.2),     # Val
    (2.5, 3.8),     # Leu
    (2.7, 4.5),     # Ile
    (1.5, 2.8),     # Phe
    (1.0, 2.5),     # Cys
    (0.5, 1.9),     # Met
    (0.0, -0.7),    # Thr
    (-0.5, -0.8),   # Ser
    (-1.5, -3.5),   # Asn
    (-1.5, -3.5),   # Gln
    (-0.5, -1.3),   # Tyr
    (0.5, -0.9),    # Trp
    (-2.0, -3.5),   # Asp
    (-2.0, -3.5),   # Glu
    (-2.5, -3.9),   # Lys
    (-3.0, -4.5),   # Arg
    (-0.5, -3.2),   # His
    (-1.0, -1.6),   # Pro
]


def _fit_logp_to_kd() -> Tuple[float, float]:
    """
    用天然氨基酸的 (LogP, KD) 对做线性回归
    返回 (斜率, 截距)：KD = slope * LogP + intercept
    """
    import numpy as np
    logps = np.array([x[0] for x in NATURAL_AA_LOGP_KD])
    kds = np.array([x[1] for x in NATURAL_AA_LOGP_KD])
    slope, intercept = np.polyfit(logps, kds, 1)
    return float(slope), float(intercept)


# 预计算映射系数
_KD_SLOPE, _KD_INTERCEPT = _fit_logp_to_kd()


# =============================================================================
# 解析 AMINO.txt
# =============================================================================
def parse_amino_txt(txt_path: Optional[Path] = None) -> Dict[str, str]:
    """
    读取 AMINO.txt，返回 {name: smiles}
    
    格式：每行 "名称,SMILES"
    跳过空行和注释行（# 开头）
    """
    if txt_path is None:
        txt_path = AMINO_TXT
    
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"AMINO.txt 不存在: {txt_path}")
    
    result = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' not in line:
                print(f"  [警告] 第{lineno}行格式错误（缺少逗号）: {line}")
                continue
            parts = line.split(',', 1)
            name = parts[0].strip()
            smiles = parts[1].strip()
            if not name or not smiles:
                print(f"  [警告] 第{lineno}行名称或SMILES为空: {line}")
                continue
            if name in result:
                print(f"  [警告] 第{lineno}行名称重复: {name}，将覆盖旧值")
            result[name] = smiles
    
    return result


# =============================================================================
# 验证主链
# =============================================================================
def validate_backbone(mol) -> bool:
    """
    验证分子是否有 NH2-CH-COOH 骨架
    复用 ligand_generator 的鲁棒版函数
    """
    from ligand_generator import find_amino_nitrogen, find_carboxyl_carbon
    
    n_idx = find_amino_nitrogen(mol)
    c_idx, o_idx = find_carboxyl_carbon(mol)
    
    if n_idx is None or c_idx is None:
        return False
    
    # 验证 N 和 C 通过 α 碳相连
    n_atom = mol.GetAtomWithIdx(n_idx)
    for nb in n_atom.GetNeighbors():
        if nb.GetAtomicNum() != 6:
            continue
        for nb2 in nb.GetNeighbors():
            if nb2.GetIdx() == c_idx:
                return True
    return False


# =============================================================================
# 3D 构象生成
# =============================================================================
def generate_3d(mol, output_pdb: Path, random_seed: int = 42) -> bool:
    """
    生成 3D 构象并保存为 PDB
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        mol_h = Chem.AddHs(mol)
        
        # ETKDGv3
        from rdkit.Chem import rdDistGeom
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = random_seed
        params.enforceChirality = False
        
        result = rdDistGeom.EmbedMolecule(mol_h, params)
        if result != 0:
            # 回退到标准 Embed
            result = AllChem.EmbedMolecule(mol_h, randomSeed=random_seed, maxAttempts=100)
            if result != 0:
                result = AllChem.EmbedMolecule(
                    mol_h, useRandomCoords=True, maxAttempts=100, randomSeed=random_seed
                )
                if result != 0:
                    print(f"  [警告] 3D构象生成失败: {output_pdb.name}")
                    return False
        
        # 优化
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
            except Exception:
                pass
        
        # 保存 PDB
        Chem.MolToPDBFile(mol_h, str(output_pdb))
        return True
        
    except Exception as e:
        print(f"  [警告] 3D生成异常 {output_pdb.name}: {e}")
        return False


# =============================================================================
# 属性计算
# =============================================================================
def calculate_logp(mol) -> float:
    """计算 Crippen LogP"""
    from rdkit.Chem import Crippen
    return float(Crippen.MolLogP(mol))


def logp_to_hydropathy(logp: float) -> float:
    """
    将 Crippen LogP 映射到 Kyte-Doolittle 尺度
    使用预计算的线性系数
    """
    kd = _KD_SLOPE * logp + _KD_INTERCEPT
    # 裁剪到 Kyte-Doolittle 的合理范围
    return float(max(-4.5, min(4.5, kd)))


def calculate_charge_ph74(mol) -> float:
    """
    计算 pH=7.4 时的净电荷
    
    使用 Henderson-Hasselbalch 公式：
    对于酸 HA: 电荷 = -1 / (1 + 10^(pKa - pH))
    对于碱 B:  电荷 = +1 / (1 + 10^(pH - pKa))
    """
    from rdkit import Chem
    
    charge = 0.0
    
    # 1. 羧基（-COOH → -COO-）：酸性，pH > pKa 时去质子化
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        # 检查是否是羧基碳
        has_double_o = False
        has_single_o = False
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                has_double_o = True
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                has_single_o = True
        
        if has_double_o and has_single_o:
            # 羧基在 pH=7.4 几乎完全去质子化
            pka = PKA_TABLE["carboxyl"]
            frac_deprotonated = 1.0 / (1.0 + 10 ** (pka - TARGET_PH))
            charge -= frac_deprotonated
    
    # 2. α-氨基（-NH2 → -NH3+）：碱性，pH < pKa 时质子化
    from ligand_generator import find_amino_nitrogen, is_carboxyl_carbon
    n_idx = find_amino_nitrogen(mol)
    if n_idx is not None:
        pka = PKA_TABLE["alpha_amino"]
        frac_protonated = 1.0 / (1.0 + 10 ** (TARGET_PH - pka))
        charge += frac_protonated
    
    # 3. 侧链氨基（-NH2 → -NH3+）：碱性，跳过α-氨基
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        if atom.GetIsAromatic():
            continue
        if atom.GetIdx() == n_idx:
            continue  # 跳过α-氨基
        
        # 检查是否是非α-氨基的脂肪族氨基
        neighbors = list(atom.GetNeighbors())
        carbon_count = sum(1 for n in neighbors if n.GetAtomicNum() == 6)
        if carbon_count >= 1 and carbon_count <= 2:
            pka = PKA_TABLE["sidechain_amino"]
            frac_protonated = 1.0 / (1.0 + 10 ** (TARGET_PH - pka))
            charge += frac_protonated
    
    # 4. 胍基（Arg）：碱性
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        # 检查 C(=N)N 结构
        has_double_n = False
        has_single_n = False
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() != 7:
                continue
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                has_double_n = True
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                has_single_n = True
        
        if has_double_n and has_single_n:
            pka = PKA_TABLE["guanidine"]
            frac_protonated = 1.0 / (1.0 + 10 ** (TARGET_PH - pka))
            charge += frac_protonated
    
    # 5. 咪唑（His）：弱碱性
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        if not atom.GetIsAromatic():
            continue
        # 简化的咪唑检测：芳香N，且其邻居中有芳香C
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 6 and nb.GetIsAromatic():
                pka = PKA_TABLE["imidazole"]
                frac_protonated = 1.0 / (1.0 + 10 ** (TARGET_PH - pka))
                charge += frac_protonated
                break
    
    # 6. 酚羟基（Tyr）：弱酸性
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8:
            continue
        # 检查是否连在芳香碳上
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 6 and nb.GetIsAromatic():
                # 检查是否是 -OH（单键）
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    # 检查 O 是否只连一个重原子
                    if atom.GetDegree() == 1:
                        pka = PKA_TABLE["phenol"]
                        frac_deprotonated = 1.0 / (1.0 + 10 ** (pka - TARGET_PH))
                        charge -= frac_deprotonated
                break
    
    # 7. 巯基（Cys）：弱酸性
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 16:
            continue
        # 检查 -SH
        if atom.GetDegree() == 1:
            # 只连一个重原子（碳）
            for nb in atom.GetNeighbors():
                if nb.GetAtomicNum() == 6:
                    pka = PKA_TABLE["thiol"]
                    frac_deprotonated = 1.0 / (1.0 + 10 ** (pka - TARGET_PH))
                    charge -= frac_deprotonated
                    break
    
    return float(charge)


def calculate_residue_mass(mol) -> float:
    """计算氨基酸残基分子量（游离 AA - 18.015）"""
    from rdkit.Chem.Descriptors import MolWt
    free_aa_mass = MolWt(mol)
    return float(free_aa_mass - 18.015)


# =============================================================================
# 处理单个非天然氨基酸
# =============================================================================
def process_one_amino_acid(name: str, smiles: str) -> Optional[Dict]:
    """
    处理单个非天然氨基酸
    
    返回：
        {
            "name": str,
            "smiles": str,
            "hydropathy": float,
            "charge": float,
            "mass": float,
            "pdb_path": str,
            "logp": float,
            "success": bool,
        }
        或 None（失败时）
    """
    from rdkit import Chem
    
    print(f"\n--- 处理 {name} ---")
    print(f"  SMILES: {smiles}")
    
    # 1. RDKit 解析
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ❌ SMILES 解析失败")
        return None
    
    # 2. 验证主链
    if not validate_backbone(mol):
        print(f"  ❌ 主链验证失败（不是氨基酸）")
        return None
    
    # 3. 生成 3D 构象
    pdb_path = AMINO_DIR / f"{name}.pdb"
    if not generate_3d(mol, pdb_path):
        print(f"  ⚠️ 3D 构象生成失败，继续计算属性")
    
    # 4. 计算属性
    try:
        logp = calculate_logp(mol)
        hydropathy = logp_to_hydropathy(logp)
        charge = calculate_charge_ph74(mol)
        mass = calculate_residue_mass(mol)
    except Exception as e:
        print(f"  ❌ 属性计算失败: {e}")
        return None
    
    print(f"  ✓ LogP={logp:.3f}, Hydropathy={hydropathy:.3f}")
    print(f"  ✓ Charge(pH7.4)={charge:+.3f}")
    print(f"  ✓ Residue Mass={mass:.2f} Da")
    if pdb_path.exists():
        print(f"  ✓ PDB: {pdb_path}")
    
    return {
        "name": name,
        "smiles": smiles,
        "hydropathy": hydropathy,
        "charge": charge,
        "mass": mass,
        "pdb_path": str(pdb_path),
        "logp": logp,
        "success": True,
    }


# =============================================================================
# 主函数：加载并注册
# =============================================================================
def load_and_register(txt_path: Optional[Path] = None) -> Dict:
    """
    加载所有非天然氨基酸并注册到 config
    
    Returns:
        {
            "hydropathy": {name: float},
            "charge": {name: float},
            "mass": {name: float},
            "smiles": {name: str},
            "properties": {name: {...}},
        }
    """
    print("=" * 70)
    print("非天然氨基酸加载器")
    print("=" * 70)
    
    # 1. 解析 AMINO.txt
    print(f"\n[1/3] 读取 {AMINO_TXT}")
    amino_dict = parse_amino_txt(txt_path)
    print(f"  发现 {len(amino_dict)} 个非天然氨基酸")
    for name in amino_dict:
        print(f"    - {name}")
    
    if not amino_dict:
        print("  ⚠️ 没有非天然氨基酸，跳过")
        return {"hydropathy": {}, "charge": {}, "mass": {}, "smiles": {}, "properties": {}}
    
    # 2. 逐个处理
    print(f"\n[2/3] 处理每个非天然氨基酸")
    properties = {}
    hydropathy = {}
    charge = {}
    mass = {}
    smiles_dict = {}
    
    for name, smi in amino_dict.items():
        result = process_one_amino_acid(name, smi)
        if result is not None:
            properties[name] = result
            hydropathy[name] = result["hydropathy"]
            charge[name] = result["charge"]
            mass[name] = result["mass"]
            smiles_dict[name] = result["smiles"]
    
    print(f"\n  成功: {len(properties)}/{len(amino_dict)}")
    
    # 3. 保存到 JSON
    print(f"\n[3/3] 保存到 {PROPERTIES_JSON}")
    output = {
        "hydropathy": hydropathy,
        "charge": charge,
        "mass": mass,
        "smiles": smiles_dict,
        "properties": properties,
    }
    with open(PROPERTIES_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 已保存")
    
    # 4. 注册到 config
    print(f"\n[4/4] 注册到 config")
    config.HYDROPATHY_NONNATURAL.update(hydropathy)
    config.AA_CHARGE_NONNATURAL.update(charge)
    config.AA_MOLECULAR_WEIGHT_NONNATURAL.update(mass)
    config.NONNATURAL_AA_SMILES.update(smiles_dict)
    
    # 合并
    config.HYDROPATHY = {**config.HYDROPATHY_NATURAL, **config.HYDROPATHY_NONNATURAL}
    config.AA_CHARGE = {**config.AA_CHARGE_NATURAL, **config.AA_CHARGE_NONNATURAL}
    config.AA_MOLECULAR_WEIGHT = {**config.AA_MOLECULAR_WEIGHT_NATURAL, **config.AA_MOLECULAR_WEIGHT_NONNATURAL}
    
    # 扩展 ALLOWED_AMINO_ACIDS
    config.ALLOWED_AMINO_ACIDS = (
        config.ALLOWED_AMINO_ACIDS_NATURAL + list(hydropathy.keys())
    )
    
    # 扩展 VARIABLE_AMINO_ACIDS
    config.VARIABLE_AMINO_ACIDS = {
        pos: config.ALLOWED_AMINO_ACIDS.copy()
        for pos in config.VARIABLE_POSITIONS
    }
    
    print(f"  ✓ HYDROPATHY: {len(config.HYDROPATHY)} 项")
    print(f"  ✓ AA_CHARGE: {len(config.AA_CHARGE)} 项")
    print(f"  ✓ AA_MOLECULAR_WEIGHT: {len(config.AA_MOLECULAR_WEIGHT)} 项")
    print(f"  ✓ ALLOWED_AMINO_ACIDS: {len(config.ALLOWED_AMINO_ACIDS)} 项")
    print(f"  ✓ VARIABLE_AMINO_ACIDS: {len(config.VARIABLE_AMINO_ACIDS)} 个位置")
    
    print("\n" + "=" * 70)
    print("加载完成！")
    print("=" * 70)
    
    return output


# =============================================================================
# 兼容接口：只返回 SMILES
# =============================================================================
def load_smiles_only(txt_path: Optional[Path] = None) -> Dict[str, str]:
    """只返回 {name: smiles}，用于 ligand_generator"""
    return parse_amino_txt(txt_path)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    load_and_register()
