# -*- coding: utf-8 -*-
"""重新生成受体 PDBQT 文件"""
from pathlib import Path
import sys
sys.path.insert(0, "/mnt/d/code/AA")
import random

from pdb_for_vina import pdb_to_pdbqt_obabel as pdb_to_pdbqt

input_pdb = Path("/mnt/d/code/AA/results/1LYZ/cleaned/cleaned.pdb")
output_pdbqt = Path("/mnt/d/code/AA/results/1LYZ/vina/vina-receptor.pdbqt")

print(f"输入: {input_pdb}")
print(f"输出: {output_pdbqt}")

pdb_to_pdbqt(input_pdb, output_pdbqt)
print("✓ 受体 PDBQT 重新生成完成")
