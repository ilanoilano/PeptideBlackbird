#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim_3: PDB 转 PDBQT 模块
功能：使用 OpenBabel 将 PDB 文件转换为 PDBQT 格式

修复：
1. 从 config.TOOLS 读取 obabel 路径，不硬编码
2. 移除 shell=True，避免注入风险
3. 使用 subprocess.run 列表传参

输入：PDB 文件路径
输出：PDBQT 文件路径
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import config

# 输出目录
OUTPUT_DIR = Path("/mnt/d/code/AA/temporary")


def pdb_to_pdbqt(pdb_path: Path, output_dir: Path = None, verbose: bool = True) -> str:
    """
    将 PDB 文件转换为 PDBQT 格式
    
    修复：使用 config.TOOLS 读取 obabel 路径，移除 shell=True
    
    Args:
        pdb_path: 输入 PDB 文件路径
        output_dir: 输出目录（默认与输入文件相同目录）
        verbose: 是否打印详细信息
    
    Returns:
        PDBQT 文件路径
    """
    pdb_path = Path(pdb_path)
    
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB 文件不存在: {pdb_path}")
    
    if output_dir is None:
        output_dir = pdb_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出文件路径
    pdbqt_path = output_dir / f"{pdb_path.stem}.pdbqt"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Sim_3: PDB 转 PDBQT")
        print(f"{'='*60}")
        print(f"输入 PDB: {pdb_path}")
        print(f"输出 PDBQT: {pdbqt_path}")
    
    # 从 config 读取 obabel 路径
    obabel_path = config.TOOLS.get("obabel", "obabel")
    
    if verbose:
        print(f"OpenBabel 路径: {obabel_path}")
    
    # 构建命令（使用列表，不使用 shell=True）
    cmd = [
        obabel_path,
        str(pdb_path),
        "-opdbqt",
        "-O", str(pdbqt_path)
    ]
    
    if verbose:
        print(f"命令: {' '.join(cmd)}")
    
    # 设置环境变量（BABEL_LIBDIR）
    env = os.environ.copy()
    babel_libdir = config.TOOLS.get("babel_libdir")
    if babel_libdir:
        env["BABEL_LIBDIR"] = babel_libdir
    
    try:
        # 执行转换（不使用 shell=True）
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if verbose:
            print(f"\nOpenBabel 返回码: {result.returncode}")
            if result.stdout:
                print(f"输出: {result.stdout.strip()}")
            if result.stderr:
                print(f"错误: {result.stderr[:200]}")
        
        if result.returncode != 0:
            raise RuntimeError(f"OpenBabel 转换失败: {result.stderr}")
        
    except FileNotFoundError:
        raise RuntimeError(
            f"OpenBabel 未找到: {obabel_path}\n"
            f"请确保 OpenBabel 已安装，或检查 config.TOOLS['obabel'] 设置"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("OpenBabel 转换超时（60秒）")
    
    # 验证输出文件
    if not pdbqt_path.exists():
        raise RuntimeError("PDBQT 文件未生成")
    
    file_size = pdbqt_path.stat().st_size
    if file_size == 0:
        raise RuntimeError("PDBQT 文件为空")
    
    # 验证格式
    with open(pdbqt_path, 'r') as f:
        content = f.read()
    
    if 'ROOT' not in content:
        raise RuntimeError("PDBQT 文件格式错误：缺少 ROOT 标签")
    
    if 'ATOM' not in content:
        raise RuntimeError("PDBQT 文件格式错误：缺少 ATOM 行")
    
    if verbose:
        print(f"\n✓ PDBQT 生成成功")
        print(f"  文件大小: {file_size} bytes")
        print(f"  包含 ROOT: True")
        print(f"  包含 ATOM: True")
        print(f"{'='*60}\n")
    
    return str(pdbqt_path)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDB 转 PDBQT 工具 (Sim_3)')
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='输入 PDB 文件路径')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='输出目录（默认与输入文件相同）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式')
    
    args = parser.parse_args()
    
    try:
        pdbqt_path = pdb_to_pdbqt(
            pdb_path=args.input,
            output_dir=args.output,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(pdbqt_path, end='')
        else:
            print(f"✓ PDBQT 文件: {pdbqt_path}")
        
        return 0
        
    except Exception as e:
        print(f"✗ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
