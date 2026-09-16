# 依赖安装指南 - OpenBabel & PDBFixer

**环境**: WSL (Windows Subsystem for Linux)  
**系统**: Ubuntu/Debian (WSL 默认)

---

## 方法一：使用 apt-get 安装（推荐）

### 1. OpenBabel 安装

```bash
# 更新包列表
sudo apt-get update

# 安装 OpenBabel
sudo apt-get install -y openbabel

# 验证安装
obabel -V
```

**预期输出**:
```
Open Babel 3.1.1 -- Oct 15 2021
```

### 2. PDBFixer 和 OpenMM 安装

**注意**: PDBFixer 依赖 OpenMM，必须同时安装

```bash
# 方式 A: 使用 conda（强烈推荐）
conda install -c conda-forge pdbfixer openmm

# 方式 B: 使用 pip
pip install pdbfixer openmm

# 验证安装
python3 -c "import pdbfixer; print(f'PDBFixer: OK')"
python3 -c "import openmm; print(f'OpenMM: {openmm.__version__}')"
```

---

## 方法二：使用 Conda 安装（如果已安装 Anaconda/Miniconda）

```bash
# 创建新环境（可选）
conda create -n mcts python=3.9
conda activate mcts

# 安装 OpenBabel
conda install -c conda-forge openbabel

# 安装 PDBFixer 和 OpenMM
conda install -c conda-forge pdbfixer openmm

# 验证
obabel -V
python3 -c "import pdbfixer; print('PDBFixer OK')"
python3 -c "import openmm; print('OpenMM OK')"
```

---

## 方法三：从源码编译安装（高级）

### OpenBabel 源码安装

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libxml2-dev \
    zlib1g-dev \
    libeigen3-dev \
    libcairo2-dev \
    libboost-all-dev

# 下载源码
cd /tmp
wget https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1-source.tar.bz2

# 解压
tar -xjf openbabel-3.1.1-source.tar.bz2
cd openbabel-3.1.1

# 编译安装
mkdir build && cd build
cmake ..
make -j4
sudo make install
sudo ldconfig

# 验证
obabel -V
```

---

## 完整一键安装脚本

创建文件 `install_deps.sh`:

```bash
#!/bin/bash
set -e

echo "========================================"
echo "  安装 OpenBabel 和 PDBFixer"
echo "========================================"

# 更新包列表
echo "[1/5] 更新包列表..."
sudo apt-get update

# 安装 OpenBabel
echo "[2/5] 安装 OpenBabel..."
sudo apt-get install -y openbabel

# 安装编译工具（用于可能的源码安装）
echo "[3/5] 安装编译工具..."
sudo apt-get install -y build-essential cmake

# 安装 PDBFixer
echo "[4/5] 安装 PDBFixer..."
if command -v conda &> /dev/null; then
    echo "检测到 Conda，使用 conda 安装..."
    conda install -c conda-forge pdbfixer -y
else
    echo "使用 pip 安装..."
    pip install pdbfixer
fi

# 验证安装
echo "[5/5] 验证安装..."
echo ""
echo "OpenBabel 版本:"
obabel -V

echo ""
echo "PDBFixer 版本:"
python3 -c "import pdbfixer; print(f'PDBFixer: {pdbfixer.__version__}')" 2>/dev/null || \
python3 -c "import pdbfixer; print('PDBFixer: 已安装')"

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
```

**运行脚本**:
```bash
cd /mnt/d/code/AA
chmod +x install_deps.sh
./install_deps.sh
```

---

## 验证安装

### 验证 OpenBabel
```bash
# 检查版本
obabel -V

# 测试转换
echo "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N" | \
    obabel -ipdb -opdbqt -p

# 检查 BABEL_LIBDIR
ls /usr/lib/openbabel/3.1.1/ 2>/dev/null || \
ls /usr/lib/x86_64-linux-gnu/openbabel/3.1.1/ 2>/dev/null || \
echo "请检查 OpenBabel 库路径"
```

### 验证 PDBFixer 和 OpenMM
```bash
# Python 导入测试
python3 << 'EOF'
import pdbfixer
import openmm
from openmm.app import PDBFile

print("✓ PDBFixer 导入成功")
print(f"  版本: {pdbfixer.__version__ if hasattr(pdbfixer, '__version__') else 'unknown'}")

print("✓ OpenMM 导入成功")
print(f"  版本: {openmm.__version__ if hasattr(openmm, '__version__') else 'unknown'}")

# 测试创建 PDBFixer
# fixer = pdbfixer.PDBFixer(filename='test.pdb')
# print("✓ PDBFixer 实例创建成功")
EOF
```

---

## 常见问题

### 问题 1: OpenBabel 找不到库文件

```bash
# 错误信息: error while loading shared libraries: libopenbabel.so.7

# 解决方案: 更新库路径
sudo ldconfig

# 或手动添加路径
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### 问题 2: PDBFixer 或 OpenMM 导入失败

```bash
# 错误信息: No module named 'pdbfixer' 或 No module named 'openmm'

# 解决方案: 重新安装（使用 conda 推荐）
conda install -c conda-forge pdbfixer openmm -y

# 或使用 pip
pip install --upgrade --force-reinstall pdbfixer openmm

# 或检查 Python 路径
python3 -c "import sys; print('\n'.join(sys.path))"
```

### 问题 3: 版本冲突

```bash
# 如果同时安装了 apt 和 conda 版本
which obabel

# 解决方案: 使用完整路径
/usr/bin/obabel -V

# 或调整 PATH
export PATH=/usr/bin:$PATH
```

### 问题 4: WSL 特定问题

```bash
# 如果在 WSL 中遇到问题，尝试更新 WSL
wsl --update

# 重启 WSL
wsl --shutdown
```

---

## 配置环境变量

添加到 `~/.bashrc`:

```bash
# OpenBabel 配置
export BABEL_LIBDIR=/usr/lib/openbabel/3.1.1
export BABEL_DATADIR=/usr/share/openbabel/3.1.1

# 如果使用 conda 安装
# export BABEL_LIBDIR=$CONDA_PREFIX/lib/openbabel/3.1.1
# export BABEL_DATADIR=$CONDA_PREFIX/share/openbabel/3.1.1
```

**应用配置**:
```bash
source ~/.bashrc
```

---

## 卸载方法

### 卸载 OpenBabel
```bash
sudo apt-get remove openbabel
sudo apt-get autoremove
```

### 卸载 PDBFixer
```bash
pip uninstall pdbfixer
# 或
conda remove pdbfixer
```

---

## 相关资源

- OpenBabel 官网: http://openbabel.org/
- PDBFixer GitHub: https://github.com/openmm/pdbfixer
- OpenMM 文档: http://docs.openmm.org/

---

**提示**: 安装完成后，请运行验证命令确保安装正确。
