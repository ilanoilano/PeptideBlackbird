#!/bin/bash
# 重建干净的 AA 环境

set -e

echo "========================================"
echo "  重建 AA 环境"
echo "========================================"
echo ""

# 获取 conda 路径
CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[1/6] 删除旧环境 AA..."
conda deactivate 2>/dev/null || true
conda remove -n AA --all -y 2>/dev/null || true

echo "[2/6] 创建新环境 AA (Python 3.9)..."
conda create -n AA python=3.9 -y

echo "[3/6] 激活环境..."
conda activate AA

echo "[4/6] 安装系统依赖 (conda)..."
conda install -c conda-forge -y \
    openbabel \
    rdkit \
    pdbfixer \
    openmm \
    matplotlib \
    pillow \
    numpy \
    pandas \
    scipy \
    tqdm

echo "[5/6] 安装 Python 包 (pip)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm

echo "[6/6] 验证安装..."
echo ""
echo "验证各组件:"
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "from rdkit import Chem; print('✓ RDKit')"
python3 -c "import openbabel; print('✓ OpenBabel')"
python3 -c "import pdbfixer; print('✓ PDBFixer')"
python3 -c "import openmm; print('✓ OpenMM')"
python3 -c "from PIL import Image; print('✓ Pillow')"
python3 -c "import matplotlib.pyplot as plt; print('✓ Matplotlib')"
python3 -c "import esm; print('✓ ESM')"

echo ""
echo "========================================"
echo "  环境重建完成！"
echo "========================================"
echo ""
echo "使用方法:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate AA"
echo ""
echo "运行项目:"
echo "  cd /mnt/d/code/AA"
echo "  python3 run_phase2.py -t 1LYZ --cold-start --n-sequences 100"
