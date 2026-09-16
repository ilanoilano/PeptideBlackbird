# 快速安装 OpenBabel 和 PDBFixer

## 方式一：一键安装（推荐）

### 在 WSL 中执行
```bash
cd /mnt/d/code/AA
bash install_deps.sh
```

### 或在 Windows 中双击
双击 `install_deps.bat`

---

## 方式二：手动安装

### 步骤 1: 安装 OpenBabel
```bash
sudo apt-get update
sudo apt-get install -y openbabel
```

### 步骤 2: 验证 OpenBabel
```bash
obabel -V
```

### 步骤 3: 安装 PDBFixer 和 OpenMM
```bash
# 方式 A: 使用 conda（推荐）
conda install -c conda-forge pdbfixer openmm

# 方式 B: 使用 pip
pip install pdbfixer openmm
```

### 步骤 4: 验证安装
```bash
python3 -c "import pdbfixer; print('PDBFixer OK')"
python3 -c "import openmm; print('OpenMM OK')"
```

---

## 配置环境变量

添加到 `~/.bashrc`:

```bash
# 查找 BABEL_LIBDIR
find /usr -name "openbabel" -type d 2>/dev/null | grep lib

# 添加到环境变量（根据实际路径修改）
export BABEL_LIBDIR=/usr/lib/openbabel/3.1.1
```

然后执行:
```bash
source ~/.bashrc
```

---

## 验证安装

```bash
# 检查 OpenBabel
obabel -V

# 检查 PDBFixer
python3 -c "import pdbfixer; print('OK')"

# 测试转换
echo "ATOM 1 N ALA A 1 10.0 10.0 10.0 1.0 0.0 N" | obabel -ipdb -opdbqt -p
```

---

## 故障排除

### OpenBabel 未找到
```bash
# 重新安装
sudo apt-get install --reinstall openbabel

# 检查路径
which obabel
```

### PDBFixer 导入失败
```bash
# 重新安装（使用 conda 推荐）
conda install -c conda-forge pdbfixer openmm -y

# 或 pip
pip uninstall pdbfixer openmm -y
pip install pdbfixer openmm

# 检查 Python 路径
python3 -c "import sys; print(sys.executable)"
```

### 权限问题
```bash
# 修复权限
sudo chown -R $(whoami) ~/.local
```

---

**安装完成后，请运行验证命令确保安装正确。**
