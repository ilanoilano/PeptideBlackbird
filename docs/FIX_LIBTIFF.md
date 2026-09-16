# 修复 libtiff / libjpeg 兼容性问题

## 问题
```
ImportError: libtiff.so.6: undefined symbol: jpeg12_write_raw_data, version LIBJPEG_8.0
```

这是 libtiff 和 libjpeg 库版本不兼容导致的 Pillow 导入失败。

---

## 解决方案

### 方案 1: 重新安装兼容版本（推荐）

在 WSL 中执行：

```bash
# 激活环境
conda activate AA

# 重新安装 libjpeg 和 libtiff
conda uninstall -y libjpeg-turbo libtiff pillow
conda clean -a -y

# 重新安装兼容版本
conda install -c conda-forge libjpeg-turbo=2.1.0 libtiff=4.4.0 -y
pip install pillow --force-reinstall --no-cache-dir
```

### 方案 2: 完全重建环境

如果方案 1 无效，重建环境：

```bash
# 删除旧环境
conda deactivate
conda remove -n AA --all -y

# 创建新环境
conda create -n AA python=3.10 -y
conda activate AA

# 按顺序安装依赖
conda install -c conda-forge libjpeg-turbo libtiff -y
pip install pillow
pip install torch fair-esm
conda install -c conda-forge openbabel rdkit pdbfixer openmm matplotlib -y
```

### 方案 3: 使用 pip 安装所有包

避免 conda 和 pip 混合使用导致的冲突：

```bash
conda activate AA

# 卸载 conda 安装的 pillow
conda uninstall -y pillow

# 使用 pip 安装
pip install pillow==9.5.0
pip install matplotlib
```

### 方案 4: 降级 Python 版本

Python 3.10 可能有兼容性问题，尝试 3.9：

```bash
conda deactivate
conda remove -n AA --all -y
conda create -n AA python=3.9 -y
conda activate AA

# 安装依赖
conda install -c conda-forge openbabel rdkit pdbfixer openmm matplotlib -y
pip install torch fair-esm
```

---

## 验证修复

```bash
conda activate AA
python3 -c "from PIL import Image; print('Pillow OK')"
python3 -c "import matplotlib.pyplot as plt; print('Matplotlib OK')"
python3 -c "import torch; print('PyTorch OK')"
```

---

## 预防措施

1. **避免 conda 和 pip 混合安装同一包**
   - 优先使用 conda 安装系统库（openbabel, rdkit 等）
   - 使用 pip 安装 Python 包（torch, fair-esm 等）

2. **安装顺序**
   ```bash
   # 1. 先安装系统库
   conda install -c conda-forge libjpeg-turbo libtiff
   
   # 2. 再安装 Python 包
   pip install pillow matplotlib
   ```

3. **使用环境文件**
   创建 `environment.yml`：
   ```yaml
   name: AA
   channels:
     - conda-forge
     - defaults
   dependencies:
     - python=3.10
     - openbabel
     - rdkit
     - pdbfixer
     - openmm
     - matplotlib
     - libjpeg-turbo
     - libtiff
     - pip
     - pip:
       - torch
       - fair-esm
   ```
   
   然后：
   ```bash
   conda env create -f environment.yml
   ```

---

## 快速修复命令

```bash
conda activate AA
conda uninstall -y pillow matplotlib
pip uninstall -y pillow matplotlib
conda install -c conda-forge libjpeg-turbo=2.1.0 libtiff=4.4.0 -y
pip install pillow==9.5.0 matplotlib --force-reinstall
```
