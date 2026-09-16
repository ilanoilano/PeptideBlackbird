# -*- coding: utf-8 -*-
"""
阶段一：PDB 预处理流水线
整合 pdb_cleaner -> (pdb_to_pockets, pdb_for_vina, receptor) 三分支
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pdb_cleaner import clean_pdb
from pdb_to_pockets import detect_pockets
from pdb_for_vina import prepare_for_vina
from receptor import prepare_af3_receptor


def run_phase1(target_name: str, pdb_filename: str = None):
    """
    运行阶段一完整流程
    
    流程：
    1. pdb_cleaner.py - 清洗 PDB
    2. 并行运行三分支：
       - pdb_to_pockets.py - 口袋检测
       - pdb_for_vina.py - Vina 受体准备
       - receptor.py - AF3 受体准备
    
    Args:
        target_name: 靶点名称（如 1LYZ）
        pdb_filename: PDB 文件名（可选，默认为 {target_name}.pdb）
    """
    print(f"\n{'#'*70}")
    print(f"# 阶段一：PDB 预处理 - {target_name}")
    print(f"{'#'*70}")
    
    # Step 1: 清洗 PDB
    print("\n" + "="*70)
    print("步骤 1/4: 清洗 PDB 结构")
    print("="*70)
    try:
        cleaned_pdb = clean_pdb(target_name, pdb_filename)
        print(f"✓ 清洗完成: {cleaned_pdb}")
    except Exception as e:
        print(f"✗ 清洗失败: {e}")
        raise
    
    # Step 2: 口袋检测
    print("\n" + "="*70)
    print("步骤 2/4: 检测结合口袋")
    print("="*70)
    try:
        pocket = detect_pockets(target_name)
        print(f"✓ 口袋检测完成")
        print(f"  中心: ({pocket['center'][0]:.3f}, {pocket['center'][1]:.3f}, {pocket['center'][2]:.3f})")
        print(f"  体积: {pocket['volume']:.1f} Å³")
    except Exception as e:
        print(f"✗ 口袋检测失败: {e}")
        raise
    
    # Step 3: Vina 受体准备
    print("\n" + "="*70)
    print("步骤 3/4: 准备 Vina 受体")
    print("="*70)
    try:
        vina_pdbqt = prepare_for_vina(target_name)
        print(f"✓ Vina 受体准备完成")
    except Exception as e:
        print(f"✗ Vina 受体准备失败: {e}")
        raise
    
    # Step 4: AF3 受体准备
    print("\n" + "="*70)
    print("步骤 4/4: 准备 AF3 受体")
    print("="*70)
    try:
        af3_pdb = prepare_af3_receptor(target_name)
        print(f"✓ AF3 受体准备完成")
    except Exception as e:
        print(f"✗ AF3 受体准备失败: {e}")
        raise
    
    # 总结
    print("\n" + "#"*70)
    print(f"# 阶段一完成: {target_name}")
    print(f"{'#'*70}")
    print("\n输出文件:")
    from config import get_target_dirs
    dirs = get_target_dirs(target_name)
    print(f"  清洗后 PDB:     {dirs['cleaned']}/cleaned.pdb")
    print(f"  口袋 JSON:      {dirs['pocket']}/pocket.json")
    print(f"  口袋 PDB:       {dirs['pocket']}/pocket-for-esmif.pdb")
    print(f"  口袋序列:       {dirs['pocket']}/pocket_seq.txt")
    print(f"  Vina 受体:      {dirs['vina']}/vina-receptor.pdbqt")
    print(f"  Vina 配置:      {dirs['vina']}/vina_config.txt")
    print(f"  AF3 受体:       {dirs['af3_receptor']}/af3-receptor.pdb")
    print("\n可以继续运行阶段二: MCTS 迭代搜索")
    
    return True


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="阶段一：PDB 预处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_phase1.py 1LYZ
  python run_phase1.py 1LYZ --pdb-file 1LYZ.ent
        """
    )
    parser.add_argument("target", help="靶点名称（如 1LYZ）")
    parser.add_argument("--pdb-file", help="PDB 文件名（默认为 {target}.pdb）")
    
    args = parser.parse_args()
    
    try:
        run_phase1(args.target, args.pdb_file)
    except Exception as e:
        print(f"\n✗ 阶段一失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
