# 🚀 快速开始指南

## 环境要求
- Windows 10/11 + WSL2
- Python 3.8+
- RDKit, PyTorch, OpenBabel, Vina

## 安装依赖
```bash
# 在 WSL 中执行
sudo apt-get update
sudo apt-get install openbabel autodock-vina
conda install -c conda-forge rdkit
pip install torch numpy pandas
```

## 一键运行（推荐）

### 方式 1: 使用 Bash 脚本（WSL 内）
```bash
cd /mnt/d/code/AA
./run_pipeline.sh 1LYZ --cold-start --n-sequences 100
```

### 方式 2: 使用 Windows 批处理
```cmd
# 在 CMD 或 PowerShell 中执行
cd D:\code\AA
run_pipeline.bat 1LYZ --cold-start --n-sequences 100
```

### 方式 3: 分步执行
```bash
# 进入项目目录
cd /mnt/d/code/AA

# 阶段一：准备受体
python3 pdb_cleaner.py 1LYZ
python3 pdb_to_pockets.py 1LYZ
python3 pdb_for_vina.py 1LYZ

# 阶段二：MCTS 搜索
python3 run_phase2.py -t 1LYZ --cold-start --n-sequences 100
```

## 常用命令

### 查看结果
```bash
cat results/1LYZ/candidates.csv
cat results/1LYZ/dataset.csv | head -20
```

### 调试模式
```bash
export MCTS_DEBUG=1
python3 run_phase2.py -t 1LYZ --cold-start --n-sequences 10
```

### 继续运行（已有模型）
```bash
python3 run_phase2.py -t 1LYZ --mcts-iter 5000 --max-iter 100000
```

### 重新训练 EGNN
```bash
python3 EGNN_1.py
python3 EGNN_23.py
python3 EGNN_4.py
```

## 文件结构
```
/mnt/d/code/AA/
├── run_pipeline.sh          # 一键运行脚本（WSL）
├── run_pipeline.bat         # 一键运行脚本（Windows）
├── WSL_COMMAND_GUIDE.md     # 完整命令清单
├── ERROR_MESSAGES_GUIDE.md  # 错误信息指南
├── QUICK_START.md           # 本文件
├── pdb_cleaner.py           # 阶段一：清理 PDB
├── pdb_to_pockets.py        # 阶段一：识别口袋
├── pdb_for_vina.py          # 阶段一：准备受体
├── run_phase2.py            # 阶段二：MCTS 搜索
├── ligand_generator.py      # 配体生成
├── vina.py                  # Vina 对接
├── egnn_predictor.py        # EGNN 预测
├── simulation.py            # MCTS 模拟
├── expansion.py             # MCTS 扩展
├── selection.py             # MCTS 选择
├── backpropagation.py       # MCTS 回溯
├── peptide_state.py         # 状态定义
├── config.py                # 配置文件
├── EGNN_1.py                # EGNN 数据准备
├── EGNN_23.py               # EGNN 训练
├── EGNN_4.py                # EGNN 评估
├── results/                 # 结果目录
│   └── 1LYZ/
│       ├── candidates.csv   # 最终候选
│       ├── dataset.csv      # 训练数据
│       └── vina/            # Vina 文件
├── checkpoints/             # 检查点
│   └── 1LYZ/
└── egnn/
    ├── raw/                 # 原始数据
    └── models/              # 训练好的模型
        └── best_model.pt
```

## 故障排除

### 问题：找不到模块
```bash
cd /mnt/d/code/AA
python3 -c "import sys; print(sys.path)"
```

### 问题：RDKit 未安装
```bash
conda install -c conda-forge rdkit
```

### 问题：OpenBabel 未安装
```bash
sudo apt-get install openbabel
```

### 问题：Vina 未安装
```bash
sudo apt-get install autodock-vina
```

### 问题：权限错误
```bash
sudo chown -R $(whoami):$(whoami) /mnt/d/code/AA
```

## 获取帮助

### 查看详细命令清单
```bash
cat WSL_COMMAND_GUIDE.md
```

### 查看错误信息指南
```bash
cat ERROR_MESSAGES_GUIDE.md
```

### 查看模块帮助
```bash
python3 run_phase2.py --help
python3 vina.py --help
python3 ligand_generator.py --help
```

## 联系方式

如有问题，请查看：
1. `ERROR_MESSAGES_GUIDE.md` - 错误信息详解
2. `WSL_COMMAND_GUIDE.md` - 完整命令清单
3. 代码中的详细报错信息（中文）

---

**提示**: 所有命令都应在 WSL 终端中执行，路径使用 `/mnt/d/code/AA` 格式。
