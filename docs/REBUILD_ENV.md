# 重建 AA 环境

## 问题
当前 AA 环境已损坏，需要重建。

## 一键重建

在 WSL 中执行：

```bash
cd /mnt/d/code/AA
bash rebuild_env.sh
```

## 手动重建

如果脚本失败，手动执行：

```bash
# 1. 删除旧环境
conda remove -n AA --all -y

# 2. 创建新环境
conda create -n AA python=3.9 -y
conda activate AA

# 3. 安装依赖（全部用 conda）
conda install -c conda-forge -y \
    openbabel rdkit pdbfixer openmm \
    matplotlib pillow numpy pandas scipy tqdm

# 4. 安装 PyPI 包
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm

# 5. 验证
python3 -c "import torch; from PIL import Image; import matplotlib.pyplot as plt; print('OK')"
```

## 重建后使用

```bash
# 激活环境
conda activate AA

# 进入项目目录
cd /mnt/d/code/AA

# 运行项目
python3 run_phase2.py -t 1LYZ --cold-start --n-sequences 100
```

## 注意事项

1. 使用 Python 3.9（比 3.10 更稳定）
2. 优先使用 conda 安装包，避免混合使用
3. PyTorch 使用 CPU 版本（避免 CUDA 问题）
