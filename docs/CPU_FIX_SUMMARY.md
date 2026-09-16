# CPU配置和对接问题修复总结

## 问题1: CPU数量未按config配置

### 问题描述
- config.py中 `VINA_CONFIG["cpu"] = 10`
- 但实际运行时使用的是 `PARALLEL_VINA_CONFIG["cpu_per_worker"] = 1`
- 导致Vina只使用1个CPU，而不是配置的10个

### 修复内容

#### 1. config.py
- 移除了 `PARALLEL_VINA_CONFIG["cpu_per_worker"]` 配置
- 添加注释说明现在统一使用 `VINA_CONFIG["cpu"]`

#### 2. vina.py
- `batch_vina_dock_parallel()`: 修改默认CPU读取逻辑
  ```python
  # 修改前
  n_cpu_per_worker = config.PARALLEL_VINA_CONFIG.get("cpu_per_worker", 1)
  
  # 修改后
  n_cpu_per_worker = config.VINA_CONFIG.get("cpu", 4)
  ```

#### 3. run_phase2.py
- `cold_start()`: 移除硬编码的 `n_cpu=1`
  ```python
  # 修改前
  result = run_vina_with_progress(..., n_cpu=1)
  
  # 修改后
  result = run_vina_with_progress(..., n_cpu=config.VINA_CONFIG.get("cpu", 4))
  ```

## 问题2: 正值结合能（对接失败）

### 问题描述
- Vina输出结合能为 **38.8 kcal/mol**（正值）
- 正值表示对接失败，分子没有正确结合
- 同时出现 `Can't kekulize mol` 错误

### 根本原因
1. **TBMB交联剂SMILES使用芳香环表示法**：`c1c(cc(cc1CBr)CBr)CBr`
2. 在创建C-S键后，芳香环无法正确kekulize
3. 导致分子结构有问题，对接失败

### 修复内容

#### 1. ligand_generator.py - 交联剂SMILES
```python
# 修改前（芳香环表示法）
"TBMB": "c1c(cc(cc1CBr)CBr)CBr"

# 修改后（Kekulé形式，明确双键）
"TBMB": "C1=CC(CBr)=CC(CBr)=C1CBr"
```

#### 2. ligand_generator.py - 增强sanitization错误处理
- 添加详细的错误日志
- 改进错误恢复逻辑
- 即使sanitization失败也尝试继续生成构象

#### 3. vina.py - 检测正值结合能
- 添加正值结合能检测
- 输出详细的警告信息
- 返回 `success=False` 而不是抛出异常

## 验证方法

### 1. 检查CPU配置
```bash
cd /mnt/d/code/AA
python -c "
import config
print(f'VINA_CONFIG[cpu] = {config.VINA_CONFIG.get(\"cpu\", 4)}')
"
```

### 2. 调试对接
```bash
python debug_docking.py -s ACPNDCGDACG -t 1LYZ
```

### 3. 运行冷启动
```bash
python run_phase2.py -t 1LYZ --cold-start --n-sequences 10
```

## 预期结果

### CPU使用
- Vina应该使用 `config.VINA_CONFIG["cpu"]` 指定的CPU数（默认10个）
- 输出中应显示：`CPU=10`

### 对接结果
- 结合能应该是负值（如 -5 到 -15 kcal/mol）
- 不应再出现 `Can't kekulize mol` 错误
- 正值结合能应该被正确检测并报告

## 如果仍然有问题

### 检查分子生成
```bash
python -c "
from ligand_generator import generate_ligand
pdbqt = generate_ligand('ACPNDCGDACG', 'TBMB', [1, 5, 9], verbose=True)
print(f'生成: {pdbqt}')
"
```

### 手动测试Vina
```bash
# 使用生成的PDBQT文件测试
vina --receptor results/1LYZ/vina/vina-receptor.pdbqt \
     --ligand temp/ligand_generator/peptide_xxx.pdbqt \
     --config results/1LYZ/vina/vina_config.txt \
     --cpu 10 \
     --exhaustiveness 4
```

### 检查受体和配置
```bash
# 检查受体文件
ls -la results/1LYZ/vina/

# 检查Vina配置
cat results/1LYZ/vina/vina_config.txt
```
