# 代码审计报告 - 模拟数据风险检查

**审计时间**: 2026-08-23  
**审计范围**: D:\code\AA 项目核心文件

---

## 🔴 已修复的严重漏洞

### 1. Vina对接失败返回0 (vina.py)

**问题**: 对接失败时返回 `0.0`，调用者可能误以为是成功结果

**修复前**:
```python
except Exception as e:
    return 0.0  # 危险！
```

**修复后**:
```python
except Exception as e:
    raise RuntimeError(f"Vina对接失败: {e}") from e
```

### 2. VinaResult返回0能量 (vina.py)

**问题**: 文件不存在或解析失败时返回 `VinaResult(0, ...)`

**修复前**:
```python
if not ligand_pdbqt.exists():
    return VinaResult(0, None, False, "...", sequence)
```

**修复后**:
```python
if not ligand_pdbqt.exists():
    raise FileNotFoundError(f"【Vina错误】配体文件不存在: {ligand_pdbqt}")
```

### 3. EGNN预测器返回None (egnn_predictor.py)

**问题**: 模型不存在时返回 `None`，调用者可能继续执行

**修复前**:
```python
def create_egnn_predictor() -> Optional[EGNNPredictor]:
    if not model_path.exists():
        return None  # 危险！
```

**修复后**:
```python
def create_egnn_predictor() -> EGNNPredictor:
    if not model_path.exists():
        raise FileNotFoundError(f"EGNN模型不存在: {model_path}")
```

---

## 🟡 需要关注的简化处理

### 1. 肽链构建 (ligand_generator.py)

**位置**: `build_peptide_with_rdkit()`

**问题**: 肽键形成是简化处理的
```python
# 形成肽键（除了第一个氨基酸）
if i > 0 and prev_carbonyl_c is not None and c_atom is not None:
    pass  # 肽键形成在构象生成后处理 ← 简化
```

**建议**: 使用RDKit的`Chem.CombineMols`和`EditableMol`正确形成肽键

### 2. 交联剂添加 (ligand_generator.py)

**位置**: `add_crosslinker()`

**问题**: 注释明确说明是简化处理
```python
# 注意：这里简化处理，实际应该删除Br并创建C-S键
```

**建议**: 实现完整的交联剂添加逻辑，包括删除Br原子

---

## ✅ 已正确实现的保护

### 1. Mock值检测 (run_phase2.py)

```python
def check_not_mock(value, name: str, expected_type=None):
    # 检查是否为随机数（在特定范围内的浮点数）
    if isinstance(value, float):
        if -12 < value < -5 and value not in [-6.0, -7.0, ...]:
            debug_print(f"【警告】{name} = {value} 可能是随机生成的mock值！", "WARNING")
```

### 2. 数据过滤 (run_phase2.py)

```python
def _update_dataset(self, results: Dict[str, float], source: str):
    # 过滤无效数据（只保留负结合能）
    if energy < 0:
        valid_results[state_key] = energy
    else:
        print(f"  [过滤] {state_key}: energy={energy} (非负值，可能是Vina失败)")
```

### 3. 详细日志输出 (所有模块)

- `simulation.py`: `[Simulation]` 前缀日志
- `egnn_predictor.py`: `[EGNNPredictor]` 前缀日志
- `run_phase2.py`: `debug_print()` 函数

---

## 📋 运行时检查清单

运行程序时，请检查以下输出：

### EGNN预测阶段
```
[EGNNPredictor] 正在加载EGNN模型: ...
[EGNNPredictor] ✓ EGNN模型加载成功
[EGNNPredictor] 开始预测: xxx.pdbqt
[EGNNPredictor] ✓ 预测完成: energy=-8.2345 kcal/mol
```

如果出现以下输出，说明使用了模拟数据：
```
[警告] EGNN预测能量 = -7.1234 可能是随机生成的mock值！
```

### Vina对接阶段
```
启动Vina对接 (CPU=1)...
Vina输出:
  ...
✓ 结合能: -9.1234 kcal/mol
```

如果出现异常，现在会抛出错误而不是返回0。

---

## 🚀 建议的测试流程

1. **测试EGNN预测**:
```bash
python -c "from egnn_predictor import create_egnn_predictor; p = create_egnn_predictor()"
```
- 如果模型不存在，应该抛出 `FileNotFoundError`
- 如果模型存在，应该输出 "✓ EGNN模型加载成功"

2. **测试Vina对接**:
```bash
python vina.py -s "ACDEFG" -t 1LYZ -v
```
- 如果受体文件不存在，应该抛出 `FileNotFoundError`
- 如果成功，应该输出真实的结合能（负值）

3. **测试完整流程**:
```bash
python run_phase2.py -t 1LYZ --cold-start
```
- 检查所有输出是否包含 `[Simulation]`, `[EGNNPredictor]` 等前缀
- 检查是否有 "【警告】" 或 "【错误】" 输出

---

## 📌 关键原则

1. **永不静默失败**: 所有错误都抛出异常，不返回默认值
2. **详细日志**: 每个步骤都有中文日志输出
3. **Mock检测**: 自动检测可能的随机数/mock值
4. **数据过滤**: 只保存有效的负结合能数据

---

**审计结论**: 已修复所有发现的模拟数据漏洞，代码现在会在失败时抛出异常而不是返回模拟值。
