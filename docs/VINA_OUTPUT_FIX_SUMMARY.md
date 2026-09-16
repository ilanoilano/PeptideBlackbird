# Vina输出解析修复总结

## 问题描述

Vina对接失败，错误信息：
```
无法从Vina输出解析结合能，可能是Vina执行失败或输出格式异常
```

## 根本原因

### 1. Vina版本差异

**Vina 1.1.2 (2011)** 输出格式：
```
   1         -2.6      0.000      0.000
   2         -2.4      1.668      3.561
```

**Vina 1.2+** 输出格式：
```
REMARK VINA RESULT:      -2.6 0.000 0.000
REMARK VINA RESULT:      -2.4 1.668 3.561
```

原来的代码只支持1.2+格式，不支持1.1.2格式。

### 2. 错误码处理

Vina返回-2表示警告（无法利用所有CPU），但代码把任何非零返回码都当作错误处理。

## 修复内容

### 1. vina.py - 支持两种输出格式

```python
# 解析结合能（支持Vina 1.1.2和1.2+格式）
for line in stdout_lines:
    # Vina 1.2+ 格式
    if 'REMARK VINA RESULT:' in line:
        ...
    # Vina 1.1.2 格式
    stripped = line.strip()
    if stripped and stripped[0].isdigit() and 'affinity' not in line.lower():
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                mode_num = int(parts[0])
                energy = float(parts[1])
                if mode_num == 1:  # 取最佳结合能
                    binding_energy = energy
                    break
            except (ValueError, IndexError):
                continue
```

### 2. vina.py - 正确处理警告返回码

```python
# returncode = 0: 成功
# returncode = -2: 警告（如无法利用所有CPU），但对接可能成功
# returncode < 0: 其他错误
if returncode < 0 and returncode != -2:
    # 真正的错误才抛出异常
    raise RuntimeError(...)

if returncode == -2:
    # 只是警告，继续解析
    print("【Vina警告】返回码-2（警告，但可能成功）")
```

## 验证

### 测试Vina输出解析

```bash
cd /mnt/d/code/AA
python3 test_vina_output.py
```

预期输出：
```
✓ 找到 Vina 1.1.2 格式: 1         -2.6      0.000      0.000
✓ 解析成功: 结合能 = -2.60 kcal/mol
```

### 测试完整对接流程

```bash
python3 debug_docking.py -s "ACPNDCGDACG" -t 1LYZ
```

## 注意事项

1. **Vina 1.1.2 vs 1.2+**: 1.1.2使用表格格式，1.2+使用REMARK格式
2. **返回码-2**: 只是警告，不代表失败
3. **结合能位置**: 1.1.2中在第2列，1.2+中在第4列

## 后续建议

考虑升级到Vina 1.2.5：
```bash
conda install -c conda-forge autodock-vina=1.2.5
```

Vina 1.2.5优势：
- 更快的搜索算法
- 更好的多核CPU利用
- 更稳定的输出格式
