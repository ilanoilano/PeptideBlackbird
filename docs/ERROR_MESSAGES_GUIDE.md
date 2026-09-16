# 错误信息指南

本文档列出了程序中所有详细的错误信息，帮助您快速定位问题。

---

## 🔴 Vina 对接错误 (vina.py)

### 1. 受体文件不存在
```
【Vina错误】受体文件不存在
  期望路径: /path/to/vina-receptor.pdbqt
  靶点名称: 1LYZ

  可能原因:
    1. 阶段一未运行，受体文件未生成
    2. 靶点名称拼写错误
    3. 受体文件被删除或移动

  解决方案:
    1. 先运行阶段一准备受体:
       python run_phase1.py -t 1LYZ
    2. 检查靶点名称是否正确
    3. 检查 results/1LYZ/vina/ 目录
```

### 2. 配体文件不存在
```
【Vina错误】配体文件不存在
  期望路径: /path/to/ligand.pdbqt

  可能原因:
    1. 配体生成失败，PDBQT文件未创建
    2. 配体生成后文件被删除
    3. 临时目录权限问题

  解决方案:
    1. 检查 ligand_generator.py 是否正常工作
    2. 检查临时目录权限
    3. 手动检查文件是否存在: ls -la /tmp/
```

### 3. Vina返回非零退出码
```
【Vina错误】Vina进程返回非零退出码
  返回码: 1
  序列: ACARNDC...
  配体: /path/to/ligand.pdbqt
  受体: /path/to/receptor.pdbqt

  最后10行输出:
    1: ...
    2: ...

  可能原因:
    1. Vina命令参数错误
    2. 配体/受体文件格式不兼容
    3. 对接盒子配置错误（中心/大小）
    4. 内存不足

  解决方案:
    1. 检查配体PDBQT格式: obabel ligand.pdbqt -opdbqt
    2. 检查受体PDBQT格式: obabel receptor.pdbqt -opdbqt
    3. 检查Vina配置文件的盒子参数
    4. 尝试减少exhaustiveness参数
```

### 4. 无法解析结合能
```
【Vina错误】无法从输出中解析结合能
  序列: ACARNDC...
  配体: /path/to/ligand.pdbqt
  受体: /path/to/receptor.pdbqt

  Vina输出内容（前500字符）:
  ...

  可能原因:
    1. Vina输出格式异常（版本不兼容？）
    2. Vina未能成功对接（配体/受体问题）
    3. Vina输出被截断

  解决方案:
    1. 检查Vina版本: vina --version
    2. 手动运行Vina查看完整输出
    3. 检查配体和受体文件是否有效
```

### 5. 对接超时
```
【Vina错误】对接超时
  超时时间: 300 秒
  序列: ACARNDC...

  可能原因:
    1. 分子过大，对接计算量过大
    2. exhaustiveness设置过高
    3. CPU资源不足
    4. Vina进程死锁（OpenMP问题）

  解决方案:
    1. 增加超时时间: timeout=600
    2. 降低exhaustiveness（默认32，尝试16或8）
    3. 使用更少的CPU核心: n_cpu=1
    4. 检查系统负载: top 或 htop
```

---

## 🔴 EGNN 预测错误 (egnn_predictor.py)

### 1. 模型文件不存在
```
【EGNN错误】模型文件不存在
  期望路径: /path/to/egnn/models/best_model.pt

  可能原因:
    1. EGNN训练未完成，模型文件未生成
    2. 模型文件被删除或移动
    3. 路径配置错误

  解决方案:
    1. 运行EGNN数据准备:
       python EGNN_1.py
    2. 运行EGNN模型训练:
       python EGNN_23.py
    3. 检查模型路径: ls -la egnn/models/
```

### 2. Checkpoint格式不正确
```
【EGNN错误】Checkpoint格式不正确
  模型路径: /path/to/best_model.pt
  可用键: ['epoch', 'loss', ...]

  可能原因:
    1. 模型文件损坏
    2. 使用不同版本的PyTorch保存/加载
    3. 训练过程中断，文件不完整

  解决方案:
    1. 删除损坏的模型文件并重新训练
    2. 检查PyTorch版本兼容性
    3. 重新运行: python EGNN_23.py
```

### 3. 模型加载失败
```
【EGNN错误】模型加载失败
  异常类型: RuntimeError
  异常信息: ...
  模型路径: /path/to/best_model.pt

  可能原因:
    1. PyTorch版本不兼容
    2. 模型文件损坏
    3. CUDA/CPU设备不匹配

  解决方案:
    1. 检查PyTorch版本: python -c 'import torch; print(torch.__version__)'
    2. 重新训练模型: python EGNN_23.py
    3. 检查CUDA可用性: python -c 'import torch; print(torch.cuda.is_available())'
```

### 4. PDBQT文件不存在
```
【EGNN错误】PDBQT文件不存在
  期望路径: /path/to/molecule.pdbqt

  可能原因:
    1. 配体生成失败
    2. 文件路径错误
    3. 临时文件被清理

  解决方案:
    1. 检查 ligand_generator.py 是否正常工作
    2. 检查临时目录权限
    3. 重新生成分子
```

### 5. 预测失败
```
【EGNN错误】预测失败
  异常类型: RuntimeError
  异常信息: ...
  文件: /path/to/molecule.pdbqt

  可能原因:
    1. PDBQT文件格式错误
    2. 原子数过多（内存不足）
    3. 模型输入维度不匹配
    4. CUDA/CPU设备错误

  解决方案:
    1. 检查PDBQT文件: cat molecule.pdbqt | head -20
    2. 检查原子数: grep -c '^ATOM' molecule.pdbqt
    3. 验证模型输入维度
    4. 检查设备可用性
```

---

## 🔴 配体生成错误 (ligand_generator.py)

### 1. RDKit未安装
```
【LigandGenerator错误】RDKit未安装

  可能原因:
    1. RDKit未安装
    2. Python环境不正确

  解决方案:
    1. 安装RDKit: conda install -c conda-forge rdkit
    2. 或: pip install rdkit
    3. 检查Python环境: which python
```

### 2. 序列为空
```
【LigandGenerator错误】序列为空

  可能原因:
    1. 序列生成失败
    2. 序列参数未正确传递
```

### 3. 未知的氨基酸
```
【LigandGenerator错误】未知的氨基酸
  未知氨基酸: X
  序列位置: 5
  完整序列: ACARNDCX...

  支持的氨基酸:
    ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

  可能原因:
    1. 序列包含非标准氨基酸
    2. 序列格式错误（包含空格或特殊字符）

  解决方案:
    1. 检查序列只包含标准氨基酸代码
    2. 移除序列中的空格和特殊字符
```

### 4. 无法解析氨基酸模板
```
【LigandGenerator错误】无法解析氨基酸模板
  氨基酸: C
  SMILES: C(CS)C(=O)O

  可能原因:
    1. SMILES格式错误
    2. RDKit版本不兼容

  解决方案:
    1. 检查AA_TEMPLATES中的SMILES格式
    2. 更新RDKit: pip install --upgrade rdkit
```

### 5. OpenBabel转换失败
```
【LigandGenerator错误】OpenBabel转换失败
  返回码: 1
  错误输出: ...
  命令: obabel temp.pdb -opdbqt -p -O output.pdbqt

  可能原因:
    1. OpenBabel未安装
    2. 临时PDB文件格式错误
    3. BABEL_LIBDIR环境变量未设置

  解决方案:
    1. 安装OpenBabel: conda install -c conda-forge openbabel
    2. 检查临时PDB文件: cat temp.pdb
    3. 设置环境变量: export BABEL_LIBDIR=/path/to/openbabel/lib
```

### 6. PDBQT文件生成失败
```
【LigandGenerator错误】PDBQT文件生成失败
  输出路径: /path/to/output.pdbqt
  文件存在: False
  文件大小: N/A

  可能原因:
    1. OpenBabel输出被拦截
    2. 磁盘空间不足
    3. 权限问题

  解决方案:
    1. 检查磁盘空间: df -h
    2. 检查目录权限: ls -la /path/to/output/
    3. 手动运行OpenBabel测试
```

---

## 🔴 Simulation 错误 (simulation.py)

### 1. EGNN模型创建失败
```
【Simulation错误】EGNN模型创建失败
  异常类型: FileNotFoundError
  异常信息: ...

  可能原因:
    1. EGNN模型文件不存在: egnn/models/best_model.pt
    2. PyTorch未安装或版本不兼容
    3. 模型文件损坏

  解决方案:
    1. 运行EGNN数据准备: python EGNN_1.py
    2. 运行EGNN模型训练: python EGNN_23.py
    3. 检查PyTorch安装: python -c 'import torch; print(torch.__version__)'
```

### 2. EGNN预测结果可能是随机数
```
【Simulation警告】EGNN预测结果可能是随机数！
  第一次预测: -7.1234
  第二次预测: -9.8765
  差异: 2.7531

  可能原因:
    1. egnn_predictor.py使用了随机数作为占位符
    2. EGNN模型未正确加载，使用了mock实现
    3. 模型权重未正确加载，使用了随机初始化

  解决方案:
    1. 检查egnn_predictor.py的predict()方法
    2. 确认模型权重已正确加载
    3. 重新训练EGNN模型
```

### 3. EGNN预测失败
```
【Simulation错误】EGNN预测失败
  异常类型: RuntimeError
  异常信息: ...
  序列: ACARNDC...
  交联剂: TBMB
  二硫键: []

  可能原因:
    1. 分子生成失败（RDKit/OpenBabel问题）
    2. EGNN模型未正确加载
    3. PDBQT文件格式错误
    4. 内存不足

  解决方案:
    1. 检查RDKit安装: python -c 'from rdkit import Chem'
    2. 检查OpenBabel安装: obabel -V
    3. 检查EGNN模型: python -c 'from egnn_predictor import create_egnn_predictor'
    4. 检查临时目录权限
```

---

## 🟡 调试技巧

### 启用调试模式
```bash
export MCTS_DEBUG=1
python run_phase2.py -t 1LYZ
```

### 检查各组件
```bash
# 检查RDKit
python -c "from rdkit import Chem; print('RDKit OK')"

# 检查OpenBabel
obabel -V

# 检查PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 检查Vina
vina --version

# 检查EGNN模型
python -c "from egnn_predictor import create_egnn_predictor; p = create_egnn_predictor(); print('EGNN OK')"
```

### 查看详细日志
所有错误信息都包含：
1. **错误类型**: 明确的错误分类
2. **错误位置**: 具体的文件和路径
3. **可能原因**: 列出常见原因
4. **解决方案**: 提供具体的修复步骤

---

**提示**: 如果遇到未列出的错误，请检查完整的堆栈跟踪信息，并查看相关文件的日志输出。
