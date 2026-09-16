#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim_2: 肽构象生成模块
功能：使用 RDKit 生成肽的 3D 构象

修复：不再重复实现，统一调用 ligand_generator.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import config
from ligand_generator import generate_ligand

OUTPUT_DIR = Path("/mnt/d/code/AA/temporary")


def generate_conformation(sequence: str, 
                          output_dir: Path = None,
                          verbose: bool = True) -> str:
    """
    生成肽的 3D 构象（PDB格式）
    
    修复：调用 ligand_generator 生成构象，然后转换为 PDB
    
    Args:
        sequence: 完整氨基酸序列
        output_dir: 输出目录
        verbose: 是否打印详细信息
    
    Returns:
        PDB 文件路径
    """
    from rdkit import Chem
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Sim_2: 肽构象生成")
        print(f"{'='*60}")
        print(f"序列: {sequence}")
        print(f"长度: {len(sequence)} 个氨基酸")
    
    # 使用 ligand_generator 生成 PDBQT（包含3D构象）
    pdbqt_path = generate_ligand(
        sequence=sequence,
        crosslinker=None,  # 无交联剂
        output_dir=output_dir,
        random_seed=42
    )
    
    # 转换为 PDB（使用 OpenBabel）
    import subprocess
    pdb_path = output_dir / f"{pdbqt_path.stem}.pdb"
    
    obabel_path = config.TOOLS.get("obabel", "obabel")
    cmd = [obabel_path, str(pdbqt_path), "-opdb", "-O", str(pdb_path)]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        raise RuntimeError(f"PDBQT 转 PDB 失败: {result.stderr}")
    
    if verbose:
        print(f"\n✓ PDB 文件: {pdb_path}")
        print(f"  文件大小: {pdb_path.stat().st_size} bytes")
        print(f"{'='*60}\n")
    
    return str(pdb_path)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='肽构象生成工具 (Sim_2)')
    parser.add_argument('-s', '--sequence', type=str, required=True,
                        help='完整氨基酸序列')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help=f'输出目录（默认: {OUTPUT_DIR}）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式')
    
    args = parser.parse_args()
    
    try:
        pdb_path = generate_conformation(
            sequence=args.sequence,
            output_dir=args.output,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(pdb_path, end='')
        else:
            print(f"✓ PDB 文件: {pdb_path}")
        
        return 0
        
    except Exception as e:
        print(f"✗ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
