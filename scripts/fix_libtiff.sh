#!/bin/bash
# 修复 libtiff / libjpeg 兼容性问题

set -e

echo "========================================"
echo "  修复 libtiff / libjpeg 兼容性"
echo "========================================"
echo ""

# 获取 conda 路径并激活环境
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate AA

echo "[1/5] 卸载冲突的包..."
conda uninstall -y pillow matplotlib libjpeg-turbo libtiff 2>/dev/null || true
pip uninstall -y pillow matplotlib 2>/dev/null || true

echo "[2/5] 清理缓存..."
conda clean -a -y
pip cache purge 2>/dev/null || true

echo "[3/5] 安装兼容版本的系统库..."
conda install -c conda-forge libjpeg-turbo=2.1.0 libtiff=4.4.0 -y

echo "[4/5] 重新安装 Pillow 和 Matplotlib..."
pip install pillow==9.5.0 matplotlib --force-reinstall --no-cache-dir

echo "[5/5] 验证安装..."
python3 -c "from PIL import Image; print('✓ Pillow OK')"
python3 -c "import matplotlib.pyplot as plt; print('✓ Matplotlib OK')"

echo ""
echo "========================================"
echo "  修复完成！"
echo "========================================"
