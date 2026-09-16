#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对接调试脚本
用于诊断Vina对接问题
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from ligand_generator import generate_ligand
from vina import get_vina_paths, run_vina_with_progress


def debug_single_dock(sequence, target_name="1LYZ", crosslinker=None, positions=None):
    """
    调试单个分子的对接过程
    
    Args:
        sequence: 氨基酸序列
        target_name: 靶点名称
        crosslinker: 交联剂类型
        positions: 交联剂位置
    """
    print("="*60)
    print("对接调试")
    print("="*60)
    print(f"序列: {sequence}")
    print(f"靶点: {target_name}")
    print(f"交联剂: {crosslinker}")
    print(f"位置: {positions}")
    print("="*60)
    
    # 1. 获取Vina路径
    print("\n[1/3] 检查Vina配置...")
    try:
        vina_paths = get_vina_paths(target_name)
        print(f"  ✓ 受体: {vina_paths['receptor']}")
        print(f"  ✓ 配置: {vina_paths['config']}")
        
        # 读取Vina配置
        with open(vina_paths['config'], 'r') as f:
            config_content = f.read()
        print(f"\n  Vina配置内容:")
        for line in config_content.strip().split('\n'):
            print(f"    {line}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return
    
    # 2. 生成分子
    print("\n[2/3] 生成分子...")
    try:
        pdbqt_path = generate_ligand(
            sequence=sequence,
            crosslinker=crosslinker,
            crosslinker_positions=positions
        )
        print(f"  ✓ PDBQT: {pdbqt_path}")
        
        # 检查PDBQT文件
        import os
        file_size = os.path.getsize(pdbqt_path)
        print(f"  文件大小: {file_size} bytes")
        
        # 读取PDBQT内容（前20行）
        with open(pdbqt_path, 'r') as f:
            lines = f.readlines()[:20]
        print(f"\n  PDBQT前20行:")
        for i, line in enumerate(lines, 1):
            print(f"    {i:2}: {line.rstrip()}")
        
        # 统计原子数
        with open(pdbqt_path, 'r') as f:
            atom_count = sum(1 for line in f if line.startswith('ATOM') or line.startswith('HETATM'))
        print(f"\n  原子数: {atom_count}")
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Vina对接
    print("\n[3/3] Vina对接...")
    print(f"  CPU: {config.VINA_CONFIG.get('cpu', 4)}")
    print(f"  Exhaustiveness: {config.VINA_CONFIG.get('exhaustiveness', 4)}")
    
    try:
        result = run_vina_with_progress(
            ligand_pdbqt=pdbqt_path,
            receptor_pdbqt=vina_paths['receptor'],
            vina_config=vina_paths['config'],
            n_cpu=config.VINA_CONFIG.get("cpu", 4),
            exhaustiveness=config.VINA_CONFIG.get("exhaustiveness", 4),
            verbose=True,
            sequence=sequence
        )
        
        print("\n" + "="*60)
        print("对接结果")
        print("="*60)
        print(f"  成功: {result.success}")
        print(f"  结合能: {result.binding_energy:.4f} kcal/mol")
        if result.error_message:
            print(f"  错误信息: {result.error_message}")
        print("="*60)
        
        # 分析结果
        if result.binding_energy > 0:
            print("\n【警告】正值结合能表示对接失败！")
            print("\n可能原因:")
            print("  1. 分子构象生成有问题")
            print("  2. 对接盒子设置不正确")
            print("  3. 配体与受体冲突")
            print("\n建议:")
            print("  1. 检查生成的PDBQT文件")
            print("  2. 使用PyMOL/Chimera可视化检查")
            print("  3. 手动运行Vina查看详细输出")
        elif result.binding_energy < -5:
            print("\n【成功】对接正常，结合能合理")
        else:
            print("\n【注意】结合能偏高，可能需要优化")
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试手动运行Vina查看输出
        print("\n" + "="*60)
        print("尝试手动运行Vina查看完整输出...")
        print("="*60)
        
        import subprocess
        vina_exe = config.TOOLS.get("vina", "vina")
        cmd = [
            vina_exe,
            "--receptor", str(vina_paths['receptor']),
            "--ligand", str(pdbqt_path),
            "--config", str(vina_paths['config']),
            "--out", str(pdbqt_path.parent / "debug_docked.pdbqt"),
            "--cpu", str(config.VINA_CONFIG.get("cpu", 4)),
            "--exhaustiveness", str(config.VINA_CONFIG.get("exhaustiveness", 4))
        ]
        
        print(f"命令: {' '.join(cmd)}")
        print("-"*60)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(f"返回码: {result.returncode}")
        print(f"\n标准输出:\n{result.stdout}")
        if result.stderr:
            print(f"\n标准错误:\n{result.stderr}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='对接调试')
    parser.add_argument('-s', '--sequence', type=str, default="ACPNDCGDACG",
                       help='氨基酸序列')
    parser.add_argument('-t', '--target', type=str, default="1LYZ",
                       help='靶点名称')
    parser.add_argument('-c', '--crosslinker', type=str, default=None,
                       help='交联剂类型（默认从config读取）')
    parser.add_argument('-p', '--positions', type=int, nargs='+', default=None,
                       help='交联剂位置（默认从config读取）')
    
    args = parser.parse_args()
    
    # 使用config默认值
    crosslinker = args.crosslinker or config.CROSSLINKER
    positions = args.positions or config.CROSSLINKER_POSITIONS
    
    debug_single_dock(args.sequence, args.target, crosslinker, positions)


if __name__ == "__main__":
    main()
