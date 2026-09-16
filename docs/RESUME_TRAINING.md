# 训练恢复功能

## 功能概述

训练过程中如果发生中断（Ctrl+C、系统崩溃等），可以从中断处继续训练，而不需要从头开始。

## 自动恢复机制

### 恢复状态文件
- 位置: `checkpoints/{target_name}/resume_state.json`
- 包含: 迭代计数、训练阶段、数据统计等

### MCTS树检查点
- 位置: `checkpoints/{target_name}/latest.json`
- 包含: 完整的MCTS树结构、节点状态

### 日志文件
- 位置: `logs/{target_name}_{timestamp}.log`
- 包含: 完整的训练日志

## 使用方法

### 正常训练（自动恢复）

```bash
# 第一次运行
python run_phase2.py -t 1LYZ

# 如果中断，再次运行相同命令即可恢复
python run_phase2.py -t 1LYZ
```

### 禁用自动恢复（从头开始）

```bash
python run_phase2.py -t 1LYZ --no-resume
```

### 重置所有状态（重新开始）

```bash
python run_phase2.py -t 1LYZ --reset
```

### 强制冷启动

```bash
python run_phase2.py -t 1LYZ --cold-start --n-sequences 100
```

## 恢复流程

### 场景1: 正常中断（Ctrl+C）

```
============================================================
训练被用户中断
============================================================
当前迭代: 15000
已保存检查点: checkpoints/1LYZ/latest.json
恢复状态: checkpoints/1LYZ/resume_state.json

下次运行将自动从中断处恢复
============================================================
```

下次运行:
```bash
python run_phase2.py -t 1LYZ
# 自动从迭代15000恢复
```

### 场景2: 系统崩溃

```bash
python run_phase2.py -t 1LYZ
# 自动检测resume_state.json并恢复
```

### 场景3: 模型文件丢失

如果EGNN模型文件丢失但恢复状态存在:
```
检测到EGNN模型缺失，但冷启动已完成标记存在...
  可能需要重新训练EGNN模型
```

需要手动重新训练EGNN模型或重新开始。

## 文件结构

```
D:/code/AA/
├── checkpoints/
│   └── 1LYZ/
│       ├── latest.json          # 最新检查点
│       ├── interrupted.json     # 中断时保存
│       ├── error.json           # 错误时保存
│       ├── final.json           # 完成时保存
│       ├── checkpoint_iter5000.json  # 定期保存
│       └── resume_state.json    # 恢复状态
├── logs/
│   ├── 1LYZ_20250825_143022.log # 训练日志1
│   └── 1LYZ_20250825_163045.log # 训练日志2
└── results/
    └── 1LYZ/
        ├── dataset.csv
        └── candidates.csv
```

## 注意事项

1. **不要手动删除检查点文件**，除非你想重新开始
2. **日志文件不会自动清理**，定期手动清理
3. **恢复时EGNN模型必须存在**，否则需要重新冷启动
4. **MCTS树可能很大**，检查点文件可能有几十MB

## 故障排除

### 恢复失败

```bash
# 检查恢复状态文件
cat checkpoints/1LYZ/resume_state.json

# 检查检查点文件
ls -lh checkpoints/1LYZ/*.json
```

### 从头开始

```bash
# 方法1: 禁用恢复
python run_phase2.py -t 1LYZ --no-resume

# 方法2: 重置所有状态
python run_phase2.py -t 1LYZ --reset
```

### 查看训练进度

```bash
# 查看最新日志
tail -f logs/1LYZ_*.log

# 查看恢复状态
cat checkpoints/1LYZ/resume_state.json
```
