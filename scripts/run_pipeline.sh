#!/bin/bash
# MCTS-Peptide 完整流程脚本
# 使用方式: ./run_pipeline.sh <target_name> [options]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
TARGET="${1:-1LYZ}"
COLD_START=false
N_SEQUENCES=100
MCTS_ITER=1000
MAX_ITER=50000

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --cold-start)
            COLD_START=true
            shift
            ;;
        --n-sequences)
            N_SEQUENCES="$2"
            shift 2
            ;;
        --mcts-iter)
            MCTS_ITER="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: ./run_pipeline.sh <target_name> [options]"
            echo ""
            echo "选项:"
            echo "  --cold-start          执行冷启动（首次运行）"
            echo "  --n-sequences N       冷启动序列数量（默认: 100）"
            echo "  --mcts-iter N         每次MCTS迭代数（默认: 1000）"
            echo "  --max-iter N          最大迭代数（默认: 50000）"
            echo "  -h, --help            显示此帮助"
            echo ""
            echo "示例:"
            echo "  ./run_pipeline.sh 1LYZ --cold-start --n-sequences 100"
            echo "  ./run_pipeline.sh 1LYZ --mcts-iter 5000 --max-iter 100000"
            exit 0
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

# 项目目录
PROJECT_DIR="/mnt/d/code/AA"
cd "$PROJECT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  MCTS-Peptide 完整流程${NC}"
echo -e "${GREEN}========================================${NC}"
echo "靶点: $TARGET"
echo "目录: $PROJECT_DIR"
echo "冷启动: $COLD_START"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: Python3 未安装${NC}"
    exit 1
fi

# 检查 RDKit
echo -e "${YELLOW}检查 RDKit...${NC}"
if ! python3 -c "from rdkit import Chem" 2>/dev/null; then
    echo -e "${RED}错误: RDKit 未安装${NC}"
    echo "运行: conda install -c conda-forge rdkit"
    exit 1
fi
echo -e "${GREEN}✓ RDKit 已安装${NC}"

# 检查 OpenBabel
echo -e "${YELLOW}检查 OpenBabel...${NC}"
if ! command -v obabel &> /dev/null; then
    echo -e "${RED}错误: OpenBabel 未安装${NC}"
    echo "运行: sudo apt-get install openbabel"
    exit 1
fi
echo -e "${GREEN}✓ OpenBabel 已安装${NC}"

# 检查 Vina
echo -e "${YELLOW}检查 Vina...${NC}"
if ! command -v vina &> /dev/null; then
    echo -e "${RED}错误: Vina 未安装${NC}"
    echo "运行: sudo apt-get install autodock-vina"
    exit 1
fi
echo -e "${GREEN}✓ Vina 已安装${NC}"

# 检查 PyTorch
echo -e "${YELLOW}检查 PyTorch...${NC}"
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${RED}错误: PyTorch 未安装${NC}"
    echo "运行: pip install torch"
    exit 1
fi
echo -e "${GREEN}✓ PyTorch 已安装${NC}"

echo ""

# ==================== 阶段一 ====================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  阶段一：受体准备${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查阶段一输出是否已存在
if [ -f "results/$TARGET/vina/vina-receptor.pdbqt" ] && [ -f "results/$TARGET/vina/vina_config.txt" ]; then
    echo -e "${YELLOW}受体文件已存在，跳过阶段一${NC}"
    echo "如需重新准备，请删除: results/$TARGET/vina/"
else
    echo -e "${YELLOW}步骤 1/3: 清理 PDB...${NC}"
    python3 pdb_cleaner.py "$TARGET"
    
    echo -e "${YELLOW}步骤 2/3: 识别结合口袋...${NC}"
    python3 pdb_to_pockets.py "$TARGET"
    
    echo -e "${YELLOW}步骤 3/3: 准备 Vina 受体...${NC}"
    python3 pdb_for_vina.py "$TARGET"
    
    echo -e "${GREEN}✓ 阶段一完成${NC}"
fi

echo ""

# ==================== 阶段二 ====================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  阶段二：MCTS 闭环搜索${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查是否需要冷启动
if [ "$COLD_START" = true ] || [ ! -f "egnn/models/best_model.pt" ]; then
    echo -e "${YELLOW}执行冷启动...${NC}"
    python3 run_phase2.py -t "$TARGET" \
        --cold-start \
        --n-sequences "$N_SEQUENCES"
else
    echo -e "${YELLOW}继续 MCTS 搜索...${NC}"
    python3 run_phase2.py -t "$TARGET" \
        --mcts-iter "$MCTS_ITER" \
        --max-iter "$MAX_ITER"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  流程完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "结果位置:"
echo "  - 候选序列: results/$TARGET/candidates.csv"
echo "  - 数据集: results/$TARGET/dataset.csv"
echo "  - 检查点: checkpoints/$TARGET/"
echo "  - EGNN模型: egnn/models/best_model.pt"
echo ""
echo "查看结果:"
echo "  cat results/$TARGET/candidates.csv"
