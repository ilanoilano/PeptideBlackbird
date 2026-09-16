# 电荷计算修复v2

## 问题

RDKit计算了Gasteiger电荷，但经过OpenBabel转换后，PDBQT文件中电荷全部变为0。

## 根本原因

OpenBabel从SDF/PDB读取时，不保留RDKit计算的Gasteiger电荷。

## 修复方案

### 新增函数: `rdkit_mol_to_pdbqt()`

直接使用RDKit生成PDBQT格式，不经过OpenBabel：

1. 从RDKit分子直接读取原子坐标
2. 从RDKit原子属性读取Gasteiger电荷 (`_GasteigerCharge`)
3. 直接写入PDBQT格式文件

### 修改函数: `mol_to_pdbqt()`

优先使用RDKit直接生成，失败时回退到OpenBabel方法。

## 测试命令

```bash
cd /mnt/d/code/AA
python3 test_charge_v2.py
```

## 预期输出

```
前15个原子电荷:
  C: +0.234
  C: -0.156
  O: -0.512
  N: +0.123
  ...

统计: 零电荷=0, 非零电荷=15

✓ 电荷计算成功！
```

## 下一步

如果电荷计算成功，运行Vina对接测试：

```bash
python3 debug_docking.py -s ACPNDCGDACG -t 1LYZ
```

预期结合能应该是负值（-5到-15 kcal/mol）。
