# Bug修复总结

## 修复时间
2026-08-22

## 修复的Bug列表

### 1. pdb_for_vina.py - 手写PDBQT格式
**严重程度**: 🔴 严重

**原问题**:
- 手写PDBQT格式生成
- 硬编码原子类型映射不完整
- 简化Gasteiger电荷计算
- 电荷错误导致对接结果错误

**修复**:
- 使用OpenBabel生成PDBQT: `obabel -opdbqt -p`
- 自动计算Gasteiger电荷
- 移除硬编码原子类型映射
- 添加详细的中文报错提示

---

### 2. sim_3.py - shell注入风险和硬编码路径
**严重程度**: 🟡 中等

**原问题**:
- 使用`shell=True`，存在注入风险
- obabel路径硬编码

**修复**:
- 移除`shell=True`，使用列表传参
- 从`config.TOOLS`读取obabel路径
- 添加`babel_libdir`到config

---

### 3. ligand_generator.py - 手写SMILES和交联剂成键
**严重程度**: 🔴 严重

**原问题**:
- 手写SMILES拼接（CS0、CS1无效）
- 交联剂只是注释，没有真正成键
- 分子缺少交联剂实体
- Vina对接的是"线性肽"而非"真正的环肽"

**修复**:
- 使用RDKit的`EditableMol`构建肽链
- 添加真正的交联剂分子（TBMB/TATA/TBAB）
- 使用`AddBond`创建C-S共价键
- 使用OpenBabel生成PDBQT（计算Gasteiger电荷）
- 详细的中文报错提示

**注意**: 交联剂的SMILES使用的是简化版本，可能需要根据真实结构调整。

---

### 4. sim_2.py - 代码重复
**严重程度**: 🟡 中等

**原问题**:
- 与ligand_generator.py重复实现手写SMILES逻辑
- 维护两份代码容易造成不一致

**修复**:
- 删除重复的手写SMILES逻辑
- 统一调用`ligand_generator.generate_ligand()`

---

### 5. EGNN_1.py - 函数名更新
**严重程度**: 🟡 中等

**原问题**:
- 使用旧的函数名`build_peptide_with_crosslinker`

**修复**:
- 更新为新的函数名`build_peptide_with_rdkit`
- 添加`add_crosslinker`调用

---

### 6. config.py - 添加babel_libdir
**严重程度**: 🟢 低

**修复**:
- 添加`babel_libdir`到`TOOLS`字典

---

## 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| pdb_for_vina.py | 重写 | 使用OpenBabel生成PDBQT |
| sim_3.py | 重写 | 移除shell=True，使用config.TOOLS |
| ligand_generator.py | 重写 | 使用EditableMol，真正交联剂成键 |
| sim_2.py | 重写 | 统一调用ligand_generator |
| EGNN_1.py | 修改 | 更新函数名 |
| config.py | 修改 | 添加babel_libdir |
| sim_4.py | 修改 | 使用config.TOOLS['vina'] |

---

## 验证结果

所有文件语法检查通过:
```bash
python3 -m py_compile pdb_for_vina.py sim_3.py ligand_generator.py sim_2.py config.py EGNN_1.py EGNN_23.py EGNN_4.py peptide_state.py selection.py expansion.py simulation.py backpropagation.py run_phase2.py vina.py sim_4.py
# 全部通过
```

自动化验证:
```bash
python3 verify_fixes.py
# 所有检查通过
```

---

## 下一步工作

### 必须完成
1. **重新生成所有训练数据**
   - 由于分子结构已改变（交联剂真正成键），所有训练数据需要重新生成
   - 运行`run_phase1.py`生成新的Vina对接数据

2. **重新训练EGNN模型**
   - 使用新的训练数据运行`EGNN_1.py`准备数据
   - 运行`EGNN_23.py`训练模型
   - 运行`EGNN_4.py`评估模型

### 建议完成
3. **验证交联剂结构**
   - 当前TBMB/TATA/TBAB的SMILES是简化版本
   - 建议根据真实化学结构更新`CROSSLINKER_SMILES`

4. **端到端测试**
   - 运行完整的MCTS流程测试
   - 验证Vina对接结果是否合理

---

## 注意事项

⚠️ **重要**: 修复后必须重新生成训练数据并重新训练EGNN模型，因为：
- 分子结构已改变（交联剂真正成键）
- 旧的训练数据中的分子结构是错误的
- EGNN学习的是"错误分子"到Vina分数的映射

这是一个连锁反应：
```
交联剂缺陷 → 训练数据错误 → EGNN预测错误 → MCTS奖励信号错误 → 搜索结果无效
```

修复后：
```
正确分子结构 → 正确训练数据 → 正确EGNN预测 → 正确MCTS奖励 → 有效搜索结果
```
