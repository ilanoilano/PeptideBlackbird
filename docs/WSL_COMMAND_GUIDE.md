# WSL 完整命令清单 - MCTS-Peptide 项目

**环境**: WSL (Windows Subsystem for Linux)  
**项目路径**: `/mnt/d/code/AA`  
**创建时间**: 2026-08-23

---

## 📋 前置准备

### 1. 进入项目目录
```bash
cd /mnt/d/code/AA
```

### 2. 检查 Python 环境
```bash
# 检查 Python 版本（需要 3.8+）
python3 --version

# 检查是否安装了必要的包
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "from rdkit import Chem; print('RDKit: OK')"
```

### 3. 检查外部工具
```bash
# 检查 OpenBabel
obabel -V

# 检查 Vina
vina --version

# 检查是否安装了 reduce（用于受体准备）
which reduce
```

---

## 🚀 阶段一：受体准备 (Phase 1)

### 步骤 1: 清理 PDB 文件
```bash
cd /mnt/d/code/AA
python3 pdb_cleaner.py 1LYZ
```

**输出位置**: `results/1LYZ/cleaned/cleaned.pdb`

### 步骤 2: 识别结合口袋
```bash
python3 pdb_to_pockets.py 1LYZ
```

**输出位置**: `results/1LYZ/pocket/pocket.json`

### 步骤 3: 准备 Vina 受体
```bash
python3 pdb_for_vina.py 1LYZ
```

**输出位置**:
- `results/1LYZ/vina/vina-receptor.pdbqt`
- `results/1LYZ/vina/vina_config.txt`

### 阶段一完整命令（一键执行）
```bash
cd /mnt/d/code/AA && \
python3 pdb_cleaner.py 1LYZ && \
python3 pdb_to_pockets.py 1LYZ && \
python3 pdb_for_vina.py 1LYZ
```

---

## 🚀 阶段二：MCTS 闭环搜索 (Phase 2)

### 方式 1: 冷启动（首次运行）
```bash
cd /mnt/d/code/AA
python3 run_phase2.py -t 1LYZ --cold-start --n-sequences 100
```

**参数说明**:
- `-t 1LYZ`: 靶点名称
- `--cold-start`: 强制冷启动，生成初始数据
- `--n-sequences 100`: 生成 100 个初始序列（可根据需要调整）

### 方式 2: 继续运行（已有 EGNN 模型）
```bash
cd /mnt/d/code/AA
python3 run_phase2.py -t 1LYZ \
    --mcts-iter 1000 \
    --val-interval 5000 \
    --max-iter 50000
```

**参数说明**:
- `--mcts-iter 1000`: 每次运行 1000 次 MCTS 迭代
- `--val-interval 5000`: 每 5000 次迭代触发验证
- `--max-iter 50000`: 最大迭代次数

### 方式 3: 调试模式（详细日志）
```bash
cd /mnt/d/code/AA
export MCTS_DEBUG=1
python3 run_phase2.py -t 1LYZ --cold-start --n-sequences 10
```

---

## 🔬 EGNN 训练（手动执行）

如果冷启动失败或需要重新训练 EGNN 模型：

### 步骤 1: 准备 EGNN 数据
```bash
cd /mnt/d/code/AA
python3 EGNN_1.py
```

**输入**: `results/1LYZ/dataset.csv`  
**输出**: `egnn/raw/*.npz`

### 步骤 2: 训练 EGNN 模型
```bash
cd /mnt/d/code/AA
python3 EGNN_23.py
```

**输出**: `egnn/models/best_model.pt`

### 步骤 3: 评估 EGNN 模型
```bash
cd /mnt/d/code/AA
python3 EGNN_4.py
```

---

## 🧪 单独测试各组件

### 测试配体生成
```bash
cd /mnt/d/code/AA
python3 ligand_generator.py -s "ACARNDCMVFLWPCG" -c TBMB -v
```

### 测试 Vina 对接
```bash
cd /mnt/d/code/AA
python3 vina.py -s "ACARNDCMVFLWPCG" -t 1LYZ -v
```

### 测试 EGNN 预测
```bash
cd /mnt/d/code/AA
python3 -c "
from egnn_predictor import create_egnn_predictor
predictor = create_egnn_predictor()
print('EGNN 模型加载成功')
"
```

### 测试 Simulation 模块
```bash
cd /mnt/d/code/AA
python3 simulation.py
```

---

## 📊 查看结果

### 查看候选结果
```bash
# 查看最终候选
cat /mnt/d/code/AA/results/1LYZ/candidates.csv

# 查看数据集
cat /mnt/d/code/AA/results/1LYZ/dataset.csv | head -20
```

### 查看检查点
```bash
# 列出所有检查点
ls -la /mnt/d/code/AA/checkpoints/1LYZ/

# 查看最新检查点
cat /mnt/d/code/AA/checkpoints/1LYZ/latest.json | head -50
```

### 查看 EGNN 训练历史
```bash
cat /mnt/d/code/AA/egnn/models/training_history.json
```

---

## 🛠️ 故障排除

### 问题 1: 找不到模块
```bash
# 确保在项目根目录
cd /mnt/d/code/AA

# 检查 Python 路径
python3 -c "import sys; print('\n'.join(sys.path))"
```

### 问题 2: RDKit 未安装
```bash
# 使用 conda 安装
conda install -c conda-forge rdkit

# 或使用 pip
pip install rdkit
```

### 问题 3: OpenBabel 未安装
```bash
# Ubuntu/Debian
sudo apt-get install openbabel

# 或使用 conda
conda install -c conda-forge openbabel
```

### 问题 4: Vina 未安装
```bash
# 下载 Vina
wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64

# 移动到 PATH
sudo mv vina_1.2.5_linux_x86_64 /usr/local/bin/vina
sudo chmod +x /usr/local/bin/vina
```

### 问题 5: 权限问题
```bash
# 修复项目权限
sudo chown -R $(whoami):$(whoami) /mnt/d/code/AA

# 修复结果目录权限
mkdir -p /mnt/d/code/AA/results/1LYZ
chmod 755 /mnt/d/code/AA/results/1LYZ
```

---

## 📝 完整工作流程示例

### 首次运行（从0开始）
```bash
# 1. 进入项目目录
cd /mnt/d/code/AA

# 2. 阶段一：准备受体
echo "=== 阶段一：准备受体 ==="
python3 pdb_cleaner.py 1LYZ
python3 pdb_to_pockets.py 1LYZ
python3 pdb_for_vina.py 1LYZ

# 3. 阶段二：冷启动
echo "=== 阶段二：冷启动 ==="
python3 run_phase2.py -t 1LYZ --cold-start --n-sequences 100

# 4. 查看结果
echo "=== 查看结果 ==="
cat results/1LYZ/candidates.csv
```

### 继续运行（已有模型）
```bash
cd /mnt/d/code/AA

# 运行更多迭代
python3 run_phase2.py -t 1LYZ \
    --mcts-iter 10000 \
    --val-interval 10000 \
    --max-iter 100000
```

### 重新训练 EGNN（如果模型效果不佳）
```bash
cd /mnt/d/code/AA

# 1. 准备数据
python3 EGNN_1.py

# 2. 训练模型
python3 EGNN_23.py

# 3. 评估模型
python3 EGNN_4.py

# 4. 继续 MCTS
python3 run_phase2.py -t 1LYZ --mcts-iter 5000
```

---

## 🔍 监控运行状态

### 查看实时日志
```bash
# 在另一个终端窗口运行
tail -f /mnt/d/code/AA/results/1LYZ/dataset.csv
```

### 查看 CPU/内存使用
```bash
# 系统监控
htop

# 或
watch -n 1 'ps aux | grep python'
```

### 查看磁盘空间
```bash
df -h /mnt/d
```

---

## 💾 备份和恢复

### 备份检查点
```bash
# 创建备份目录
mkdir -p /mnt/d/code/AA/backups/$(date +%Y%m%d)

# 备份检查点
cp -r /mnt/d/code/AA/checkpoints/1LYZ/* /mnt/d/code/AA/backups/$(date +%Y%m%d)/

# 备份 EGNN 模型
cp /mnt/d/code/AA/egnn/models/best_model.pt /mnt/d/code/AA/backups/$(date +%Y%m%d)/
```

### 恢复检查点
```bash
# 从备份恢复
cp /mnt/d/code/AA/backups/20260823/checkpoint_iter10000.json \
   /mnt/d/code/AA/checkpoints/1LYZ/latest.json
```

---

## ⚡ 性能优化

### 使用多核 CPU
```bash
# 查看可用 CPU 核心数
nproc

# 在 config.py 中设置并行参数
# PARALLEL_VINA_CONFIG["num_workers"] = 8
```

### 减少内存使用
```bash
# 清理临时文件
rm -rf /tmp/mcts_simulation/*
rm -rf /mnt/d/code/AA/temp/*
```

---

## 📞 获取帮助

### 查看帮助信息
```bash
# 阶段二帮助
python3 run_phase2.py --help

# Vina 帮助
python3 vina.py --help

# 配体生成帮助
python3 ligand_generator.py --help
```

### 查看错误日志
```bash
# 查看最近的错误
python3 run_phase2.py -t 1LYZ 2>&1 | tee run.log
grep -i "error\|错误" run.log
```

---

**提示**: 所有命令都应在 WSL 终端中执行，确保路径使用 `/mnt/d/` 格式而非 `D:\`。
