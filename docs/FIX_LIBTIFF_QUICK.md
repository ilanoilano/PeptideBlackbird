# 快速修复 libtiff / libjpeg 问题

## 问题原因
`conda activate` 需要先初始化 shell。

## 快速修复（手动执行）

### 步骤 1: 初始化 conda（如果还没做过）
```bash
conda init bash
```
然后 **关闭并重新打开 WSL 终端**。

### 步骤 2: 修复库冲突
```bash
# 激活环境
conda activate AA

# 卸载冲突的包
conda uninstall -y pillow matplotlib libjpeg-turbo libtiff
pip uninstall -y pillow matplotlib

# 清理缓存
conda clean -a -y

# 安装兼容版本
conda install -c conda-forge libjpeg-turbo=2.1.0 libtiff=4.4.0 -y
pip install pillow==9.5.0 matplotlib --force-reinstall

# 验证
python3 -c "from PIL import Image; print('Pillow OK')"
python3 -c "import matplotlib.pyplot as plt; print('Matplotlib OK')"
```

---

## 或者使用 source 方式（不依赖 conda init）

```bash
# 直接 source conda.sh
source ~/miniconda3/etc/profile.d/conda.sh

# 然后激活环境
conda activate AA

# 执行修复
conda uninstall -y pillow matplotlib libjpeg-turbo libtiff
pip uninstall -y pillow matplotlib
conda install -c conda-forge libjpeg-turbo=2.1.0 libtiff=4.4.0 -y
pip install pillow==9.5.0 matplotlib --force-reinstall
```

---

## 最简方案：直接 pip 重装

如果上述方法太复杂，直接：

```bash
# 先确保环境已激活（如果 conda activate 失败，用 source 方式）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate AA

# 强制重装 pillow
pip install pillow==9.5.0 --force-reinstall --no-deps

# 测试
python3 -c "from PIL import Image; print('OK')"
```

---

## 一劳永逸：创建新的干净环境

如果修复太麻烦，直接重建环境：

```bash
# 删除旧环境
conda deactivate
conda remove -n AA --all -y

# 创建新环境
conda create -n AA python=3.9 -y

# 激活
conda activate AA

# 安装所有依赖（只用 conda，避免混合）
conda install -c conda-forge openbabel rdkit pdbfixer openmm matplotlib pillow -y

# 安装 pip 专属包
pip install torch fair-esm
```
