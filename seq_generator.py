#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列生成器 (seq_generator.py)
功能：根据模板生成随机序列，只替换 'x'/'X'/'_' 位置的氨基酸

输入：
- config.PEPTIDE_TEMPLATE: 肽序列模板
- config.FIXED_POSITIONS: 固定位置映射
- config.VARIABLE_AMINO_ACIDS: 每个可变位置允许的氨基酸

输出：
- 完整序列（无占位符）
"""

import random
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import config


def generate_full_sequence(partial_sequence: Optional[str] = None) -> str:
    """
    生成完整序列，只替换 'x'/'X'/'_' 位置的氨基酸
    
    Args:
        partial_sequence: 部分序列（可能包含占位符）
                         如果为None，使用config中的模板
    
    Returns:
        完整序列（无占位符）
    
    示例：
        "AC_____C______CG" → "ACARNDCMVFLWPCG"
        "ACARNDC______CG" → "ACARNDCMVFLWPCG"（保留已填充位置）
    """
    if partial_sequence is None:
        partial_sequence = config.PEPTIDE_TEMPLATE
    
    seq_list = list(partial_sequence)
    
    for i, char in enumerate(seq_list):
        # 只替换占位符
        if char in ['x', 'X', '_']:
            # 检查是否是固定位置（理论上不应该有占位符在固定位置）
            if i in config.FIXED_POSITIONS:
                seq_list[i] = config.FIXED_POSITIONS[i]
            else:
                # 从该位置允许的氨基酸中随机选择
                allowed_aas = config.VARIABLE_AMINO_ACIDS.get(i, config.ALLOWED_AMINO_ACIDS)
                seq_list[i] = random.choice(allowed_aas)
    
    return ''.join(seq_list)


def generate_multiple_sequences(n_sequences: int,
                                 partial_sequence: Optional[str] = None) -> List[str]:
    """
    生成多个随机序列

    Args:
        n_sequences: 需要生成的序列数量
        partial_sequence: 部分序列模板（默认使用config模板）

    Returns:
        序列列表
    """
    sequences = []
    for _ in range(n_sequences):
        seq = generate_full_sequence(partial_sequence)
        sequences.append(seq)
    return sequences


def generate_n_random_fills(partial_sequence: str, n: int) -> List[str]:
    """
    【新增】为部分序列生成N个不同的随机填充完整序列

    这是新流程的核心函数：
    - 保留已填充的氨基酸
    - 对每个占位符独立随机选择氨基酸
    - 生成N个不同的完整序列

    Args:
        partial_sequence: 部分序列（如 "AC_D_C___CG"）
        n: 需要生成的序列数量

    Returns:
        N个不同的完整序列列表

    示例：
        Input:  "AC_D_C___CG", n=3
        Output: ["ACADACDEFCG", "ACPDGCHQWCG", "ACRDVCMLKCG"]
    """
    if not partial_sequence:
        raise ValueError("partial_sequence 不能为空")

    sequences = []
    attempts = 0
    max_attempts = n * 10  # 防止无限循环

    while len(sequences) < n and attempts < max_attempts:
        attempts += 1
        seq = generate_full_sequence(partial_sequence)

        # 确保不重复
        if seq not in sequences:
            sequences.append(seq)

    return sequences


def generate_sequences_to_file(n_sequences: int,
                                output_path: Path,
                                partial_sequence: Optional[str] = None):
    """
    生成序列并保存到文件
    
    Args:
        n_sequences: 序列数量
        output_path: 输出文件路径
        partial_sequence: 部分序列模板
    """
    sequences = generate_multiple_sequences(n_sequences, partial_sequence)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Generated sequences\n")
        f.write("# Template: {}\n".format(config.PEPTIDE_TEMPLATE))
        f.write("# Crosslinker: {}\n".format(config.CROSSLINKER))
        f.write("sequence\n")
        for seq in sequences:
            f.write(f"{seq}\n")
    
    print(f"✓ 生成 {n_sequences} 个序列，保存至: {output_path}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='序列生成器')
    parser.add_argument('-n', '--num', type=int, default=100,
                       help='生成序列数量（默认100）')
    parser.add_argument('-o', '--output', type=Path, 
                       default=config.BASE_DIR / "sequences.txt",
                       help='输出文件路径')
    parser.add_argument('-t', '--template', type=str, default=None,
                       help='序列模板（默认使用config）')
    
    args = parser.parse_args()
    
    generate_sequences_to_file(args.num, args.output, args.template)


if __name__ == "__main__":
    main()
