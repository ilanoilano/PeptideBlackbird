# 修改后测试命令

## 修改内容总结

1. **修改点1**: `generate_3d_conformation()` 中RDKit计算Gasteiger电荷
2. **修改点2**: `mol_to_pdbqt()` 中删除obabel的 `-p` 参数
3. **修改点3**: `mol_to_pdbqt()` 使用SDF格式保留键序信息，解决kekulize问题

## 测试命令

### 1. 测试电荷计算和kekulize修复

```bash
cd /mnt/d/code/AA
python3 test_charge_fix.py
```

**预期输出**:
```
✓ Gasteiger电荷计算成功！
  示例电荷值:
    原子1: 0.234
    原子2: -0.156
    ...

✓ 电荷已正确计算
```

### 2. 测试完整对接流程

```bash
python3 debug_docking.py -s "ACPNDCGDACG" -t 1LYZ
```

**预期输出**:
```
[3/3] Vina对接...
  CPU: 10
  Exhaustiveness: 2

启动Vina对接 (CPU=10, exhaustiveness=2)...
...
✓ 结合能: -8.5 kcal/mol  (负值表示成功！)
```

### 3. 测试生成的PDBQT文件

```bash
# 检查最新的PDBQT文件
cd /mnt/d/code/AA/temp/ligand_generator
ls -lt *.pdbqt | head -1

# 查看电荷（第9列应该是非零值）
head -20 peptide_*.pdbqt | grep "ATOM"
```

**预期输出** (第9列应该有非零值):
```
ATOM      1  C   UNK A   2      -0.285   2.904  -7.114  0.00  0.00    +0.234 C
ATOM      2  C   UNK A   2      -1.127   3.940  -6.312  0.00  0.00    -0.156 C
...
```

### 4. 手动运行Vina测试

```bash
cd /mnt/d/code/AA

# 生成测试分子
python3 -c "
from ligand_generator import generate_ligand
pdbqt = generate_ligand('ACPNDCGDACG', 'TBMB', [1, 5, 9])
print(f'生成: {pdbqt}')
"

# 运行Vina
vina --receptor results/1LYZ/vina/vina-receptor.pdbqt \
     --ligand temp/ligand_generator/peptide_*.pdbqt \
     --config results/1LYZ/vina/vina_config.txt \
     --cpu 10 \
     --exhaustiveness 2
```

**预期输出**:
```
1         -8.5      0.000      0.000
```

### 5. 运行冷启动

```bash
python3 run_phase2.py -t 1LYZ --cold-start --n-sequences 10
```

**预期输出**:
```
[2/5] Vina对接（获取真实分数）...
[1/10] PeptideState(...)
  ✓ 结合能: -7.2
[2/10] PeptideState(...)
  ✓ 结合能: -8.1
...
```

## 故障排除

### 如果电荷仍然是0

```bash
# 检查RDKit是否能计算电荷
python3 -c "
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles('CCO')
mol = Chem.AddHs(mol)
AllChem.ComputeGasteigerCharges(mol)

for atom in mol.GetAtoms():
    print(f'{atom.GetSymbol()}: {atom.GetDoubleProp(\"_GasteigerCharge\")}')
"
```

### 如果还有kekulize错误

```bash
# 检查SDF转换是否成功
python3 -c "
from rdkit import Chem
from ligand_generator import build_peptide_with_rdkit, add_crosslinker

mol = build_peptide_with_rdkit('ACPNDCGDACG')
mol = add_crosslinker(mol, 'TBMB', [1, 5, 9], 'ACPNDCGDACG')

# 测试SDF写入
writer = Chem.SDWriter('test.sdf')
writer.write(mol)
writer.close()
print('SDF写入成功')
"

# 检查SDF文件
obabel test.sdf -opdbqt -xl -O test.pdbqt
grep "ATOM" test.pdbqt | head -5
```

## 关键检查点

1. **电荷非零**: PDBQT第9列应该是非零值（如+0.234, -0.156）
2. **结合能负值**: Vina输出应该是-5到-15 kcal/mol
3. **无kekulize错误**: 不应该出现"Can't kekulize mol"错误
