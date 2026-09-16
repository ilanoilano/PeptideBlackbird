#!/bin/bash
# 一键安装 OpenBabel 和 PDBFixer
# 使用方式: ./install_deps.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  安装 OpenBabel 和 PDBFixer${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检测系统
if ! grep -q "Ubuntu\|Debian" /etc/os-release 2>/dev/null; then
    echo -e "${YELLOW}警告: 未检测到 Ubuntu/Debian 系统${NC}"
    echo "此脚本针对 Ubuntu/Debian 优化"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 步骤 1: 更新包列表
echo -e "${YELLOW}[1/5] 更新包列表...${NC}"
sudo apt-get update -qq

# 步骤 2: 安装 OpenBabel
echo -e "${YELLOW}[2/5] 安装 OpenBabel...${NC}"
if ! command -v obabel &> /dev/null; then
    sudo apt-get install -y -qq openbabel
    echo -e "${GREEN}✓ OpenBabel 安装完成${NC}"
else
    echo -e "${GREEN}✓ OpenBabel 已安装${NC}"
fi

# 步骤 3: 安装编译工具
echo -e "${YELLOW}[3/5] 安装编译工具...${NC}"
sudo apt-get install -y -qq build-essential cmake wget

# 步骤 4: 安装 PDBFixer 和 OpenMM
echo -e "${YELLOW}[4/5] 安装 PDBFixer 和 OpenMM...${NC}"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: Python3 未安装${NC}"
    exit 1
fi

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}安装 pip...${NC}"
    sudo apt-get install -y -qq python3-pip
fi

# 安装 PDBFixer 和 OpenMM
PDBFIXER_OK=false
OPENMM_OK=false

python3 -c "import pdbfixer" 2>/dev/null && PDBFIXER_OK=true
python3 -c "import openmm" 2>/dev/null && OPENMM_OK=true

if [ "$PDBFIXER_OK" = true ] && [ "$OPENMM_OK" = true ]; then
    echo -e "${GREEN}✓ PDBFixer 和 OpenMM 已安装${NC}"
else
    if command -v conda &> /dev/null; then
        echo "使用 conda 安装 PDBFixer 和 OpenMM..."
        conda install -c conda-forge pdbfixer openmm -y
    else
        echo -e "${YELLOW}警告: 未检测到 Conda${NC}"
        echo "PDBFixer 和 OpenMM 推荐使用 conda 安装"
        echo "尝试使用 pip 安装..."
        pip3 install pdbfixer openmm
    fi
    
    # 验证安装
    if python3 -c "import pdbfixer; import openmm" 2>/dev/null; then
        echo -e "${GREEN}✓ PDBFixer 和 OpenMM 安装完成${NC}"
    else
        echo -e "${RED}✗ PDBFixer 或 OpenMM 安装失败${NC}"
        echo "建议手动安装:"
        echo "  conda install -c conda-forge pdbfixer openmm"
        exit 1
    fi
fi

# 步骤 5: 验证安装
echo -e "${YELLOW}[5/5] 验证安装...${NC}"
echo ""

# 验证 OpenBabel
echo -n "OpenBabel: "
if command -v obabel &> /dev/null; then
    OBABEL_VERSION=$(obabel -V 2>&1 | head -1)
    echo -e "${GREEN}$OBABEL_VERSION${NC}"
else
    echo -e "${RED}未找到${NC}"
fi

# 验证 PDBFixer
echo -n "PDBFixer:  "
if python3 -c "import pdbfixer" 2>/dev/null; then
    PDBFIXER_VERSION=$(python3 -c "import pdbfixer; print(pdbfixer.__version__ if hasattr(pdbfixer, '__version__') else 'installed')" 2>/dev/null)
    echo -e "${GREEN}$PDBFIXER_VERSION${NC}"
else
    echo -e "${RED}未找到${NC}"
fi

# 验证 OpenMM
echo -n "OpenMM:    "
if python3 -c "import openmm" 2>/dev/null; then
    OPENMM_VERSION=$(python3 -c "import openmm; print(openmm.__version__ if hasattr(openmm, '__version__') else 'installed')" 2>/dev/null)
    echo -e "${GREEN}$OPENMM_VERSION${NC}"
else
    echo -e "${RED}未找到${NC}"
fi

# 查找 BABEL_LIBDIR
echo -n "BABEL_LIBDIR: "
BABEL_LIBDIR=$(find /usr -name "openbabel" -type d 2>/dev/null | grep -E "lib|lib64" | head -1)
if [ -n "$BABEL_LIBDIR" ]; then
    echo -e "${GREEN}$BABEL_LIBDIR${NC}"
    
    # 添加到 .bashrc
    if ! grep -q "BABEL_LIBDIR" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# OpenBabel 配置" >> ~/.bashrc
        echo "export BABEL_LIBDIR=$BABEL_LIBDIR" >> ~/.bashrc
        echo -e "${GREEN}✓ 已添加 BABEL_LIBDIR 到 ~/.bashrc${NC}"
    fi
else
    echo -e "${YELLOW}未找到，可能需要手动配置${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  安装完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "使用以下命令验证:"
echo "  obabel -V"
echo "  python3 -c 'import pdbfixer; print(\"OK\")'"
echo ""
echo "如果 PDBFixer 导入失败，请尝试:"
echo "  source ~/.bashrc"
echo "  或重启终端"
