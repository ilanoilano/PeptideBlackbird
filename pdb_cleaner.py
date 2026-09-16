# -*- coding: utf-8 -*-
"""
PDB 结构清洗模块
功能：去水、补缺失残基、加氢
输出：results/[target_name]/cleaned/cleaned.pdb
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import PDB_DIR, get_target_dirs, TOOLS


def run_pdbfixer(input_pdb: Path, output_pdb: Path) -> bool:
    """
    使用 PDBFixer 清洗 PDB 文件
    
    步骤：
    1. 去除水分子
    2. 添加缺失的重原子
    3. 添加氢原子（pH 7.0）
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        print("【需要安装】pdbfixer 和 openmm")
        print("请运行: conda install -c conda-forge pdbfixer openmm")
        return False
    
    print(f"  加载 PDB: {input_pdb}")
    fixer = PDBFixer(str(input_pdb))
    
    # 1. 查找并移除非标准残基（包括水）
    print("  查找非标准残基...")
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        print(f"    发现 {len(fixer.nonstandardResidues)} 个非标准残基")
        fixer.replaceNonstandardResidues()
    
    # 2. 移除水分子
    print("  移除水分子...")
    fixer.removeHeterogens(keepWater=False)
    
    # 3. 查找缺失残基
    # 3. 查找缺失残基（新版本 pdbfixer 不需要 addMissingResidues）
    print("  查找缺失残基...")
    fixer.findMissingResidues()
    if fixer.missingResidues:
        print(f"    发现 {len(fixer.missingResidues)} 个缺失残基")
        chains = list(fixer.topology.chains())
        keys_to_remove = []
        for key in fixer.missingResidues:
            chain_idx, res_idx = key
            if res_idx == 0 or res_idx >= len(list(chains[chain_idx].residues())):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del fixer.missingResidues[key]
        # 新版本 pdbfixer 用 addMissingAtoms 自动处理缺失残基
        # fixer.addMissingResidues()  # ← 删除或注释掉这行
    # 直接跳到 addMissingAtoms
    fixer.findMissingAtoms()
    
    # 4. 查找缺失原子
    print("  查找缺失原子...")
    fixer.findMissingAtoms()
    if fixer.missingAtoms:
        print(f"    添加 {len(fixer.missingAtoms)} 个缺失原子")
        fixer.addMissingAtoms()
    
    # 5. 添加氢原子（pH 7.0）
    print("  添加氢原子 (pH 7.0)...")
    fixer.addMissingHydrogens(7.0)
    
    # 6. 保存结果
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(str(output_pdb), 'w'))
    print(f"  ✓ 清洗完成: {output_pdb}")
    
    return True


def clean_pdb(target_name: str, pdb_filename: str = None) -> Path:
    """
    清洗指定靶点的 PDB 文件
    
    Args:
        target_name: 靶点名称（用于创建输出目录）
        pdb_filename: PDB文件名，默认为 {target_name}.pdb
    
    Returns:
        清洗后的 PDB 文件路径
    """
    if pdb_filename is None:
        # 尝试常见扩展名
        for ext in [".pdb", ".ent"]:
            candidate = PDB_DIR / f"{target_name}{ext}"
            if candidate.exists():
                pdb_filename = f"{target_name}{ext}"
                break
        else:
            raise FileNotFoundError(f"PDB 目录中未找到 {target_name}.pdb 或 {target_name}.ent")
    
    input_pdb = PDB_DIR / pdb_filename
    if not input_pdb.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_pdb}")
    
    print(f"\n{'='*60}")
    print(f"清洗 PDB: {target_name}")
    print(f"{'='*60}")
    print(f"输入: {input_pdb}")
    
    # 创建输出目录
    dirs = get_target_dirs(target_name)
    output_pdb = dirs["cleaned"] / "cleaned.pdb"
    
    # 运行清洗
    success = run_pdbfixer(input_pdb, output_pdb)
    
    if success:
        # 统计信息
        atom_count = count_atoms(output_pdb)
        print(f"  原子数: {atom_count}")
        return output_pdb
    else:
        raise RuntimeError("PDB 清洗失败")


def count_atoms(pdb_file: Path) -> int:
    """统计 PDB 文件中的原子数"""
    count = 0
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                count += 1
    return count


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="清洗 PDB 结构文件")
    parser.add_argument("target", help="靶点名称（如 1LYZ）")
    parser.add_argument("--pdb-file", help="PDB 文件名（默认为 {target}.pdb）")
    
    args = parser.parse_args()
    
    try:
        output = clean_pdb(args.target, args.pdb_file)
        print(f"\n✓ 成功: {output}")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
