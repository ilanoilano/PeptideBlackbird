# 项目文件结构

## 核心代码文件

### MCTS模块
- `backpropagation.py` - MCTS反向传播
- `expansion.py` - MCTS扩展引擎
- `peptide_state.py` - 肽状态定义
- `run_phase2.py` - 阶段二主程序（MCTS+EGNN）
- `selection.py` - PUCT选择器
- `simulation.py` - MCTS模拟引擎

### 分子生成与对接
- `ligand_generator.py` - 配体生成器（序列→PDBQT）
- `pdb_for_vina.py` - PDB文件处理
- `pdb_cleaner.py` - PDB清理工具
- `pdb_to_pockets.py` - 口袋检测
- `receptor.py` - 受体处理
- `regenerate_receptor.py` - 受体重新生成
- `vina.py` - Vina/GNINA对接集成

### 序列生成
- `seq_generator.py` - 序列生成器

### EGNN模块
- `EGNN_1.py` - EGNN数据准备
- `EGNN_23.py` - EGNN训练
- `EGNN_4.py` - EGNN评估
- `egnn_predictor.py` - EGNN预测器

### 阶段一
- `run_phase1.py` - 阶段一主程序（受体准备）

### 模拟模块（遗留）
- `sim_2.py`, `sim_3.py`, `sim_4.py` - 模拟模块（已集成到vina.py）

### 配置
- `config.py` - 全局配置
- `requirement.txt` - 依赖列表

### 调试工具
- `debug_docking.py` - 对接调试脚本

## 文档（docs/）

### 修复记录
- `BUGFIX_SUMMARY.md` - Bug修复总结
- `CHARGE_FIX_V2.md` - 电荷计算修复v2
- `CPU_FIX_SUMMARY.md` - CPU配置修复
- `VINA_OUTPUT_FIX_SUMMARY.md` - Vina输出解析修复
- `SUCCESS_SUMMARY.md` - 修复成功总结

### 使用指南
- `QUICK_START.md` - 快速开始
- `TEST_COMMANDS.md` - 测试命令
- `RESUME_TRAINING.md` - 训练恢复指南
- `INSTALL_DEPENDENCIES.md` - 依赖安装
- `INSTALL_QUICK.md` - 快速安装
- `ERROR_MESSAGES_GUIDE.md` - 错误信息指南

### 其他
- `CODE_AUDIT_REPORT.md` - 代码审计报告
- `FIX_LIBTIFF.md` / `FIX_LIBTIFF_QUICK.md` - libtiff修复
- `REBUILD_ENV.md` - 环境重建
- `WSL_COMMAND_GUIDE.md` - WSL命令指南

## 生成的目录

```
D:/code/AA/
├── checkpoints/          # MCTS检查点
│   └── {target_name}/
├── docs/                 # 文档
├── egnn/                 # EGNN模型和数据
│   ├── models/
│   └── raw/
├── logs/                 # 训练日志
├── PDB/                  # PDB文件
├── results/              # 结果
│   └── {target_name}/
├── temp/                 # 临时文件
│   └── ligand_generator/
└── __pycache__/          # Python缓存
```

## 已删除的文件

### 测试文件
- `test_charge_fix.py`
- `test_charge_v2.py`
- `test_vina_output.py`
- `check_charges.py`

### 文档（已移动到docs/）
- `CHARGE_FIX_V2.md`
- `CPU_FIX_SUMMARY.md`
- `RESUME_TRAINING.md`
- `SUCCESS_SUMMARY.md`
- `TEST_COMMANDS.md`
- `VINA_OUTPUT_FIX_SUMMARY.md`
