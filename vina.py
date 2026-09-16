
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vina对接集成模块 (vina.py)
功能：序列 → 3D构象 → PDBQT → GNINA 打分

【完全使用 GNINA 替代 AutoDock Vina】
- GNINA 基于 CNN 深度学习，打分更准确
- 支持 --no_gpu 在 CPU 上运行
- 所有原有接口保持不变（vina_dock, run_vina_with_progress, batch_vina_dock）
"""

import os
import sys
import re
import random
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import get_target_dirs
sys.path.insert(0, str(Path(__file__).parent))

import config
from ligand_generator import generate_ligand
from config import VINA_VALIDATION


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class VinaResult:
    """对接结果"""
    binding_energy: float = 0.0       # kcal/mol
    output_file: Optional[Path] = None
    success: bool = True
    error_message: str = ""
    sequence: str = ""
    validation_passed: bool = True
    validation_reason: str = ""
    # GNINA 额外输出
    cnnscore: float = 0.0
    cnnaffinity: float = 0.0
    cnnvariance: float = 0.0


# =============================================================================
# 口袋边界验证函数
# =============================================================================
def get_pocket_center(target_name: str):
    """
    根据target_name读取vina_config.txt，获取口袋中心坐标
    :param target_name: 靶点名称，如 "6Y9A"
    :return: list[float] [cx, cy, cz]; 解析失败返回 None
    """
    dirs = get_target_dirs(target_name)
    cfg_path: Path = dirs["vina"] / "vina_config.txt"

    if not cfg_path.exists():
        return None

    cx = cy = cz = None
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith("#"):
                continue

            m = re.match(r"center_x\s*=\s*([-\d.]+)", line_strip)
            if m:
                cx = float(m.group(1))
            m = re.match(r"center_y\s*=\s*([-\d.]+)", line_strip)
            if m:
                cy = float(m.group(1))
            m = re.match(r"center_z\s*=\s*([-\d.]+)", line_strip)
            if m:
                cz = float(m.group(1))

    if cx is not None and cy is not None and cz is not None:
        return [cx, cy, cz]
    return None


def load_pocket_boundaries(target_name: str) -> List[Dict]:
    """从 pocket.json 加载所有口袋的边界信息"""
    dirs = config.get_target_dirs(target_name)
    pocket_json = dirs["pocket"] / "pocket.json"

    if not pocket_json.exists():
        return []

    try:
        with open(pocket_json, 'r') as f:
            data = json.load(f)

        if "all_pockets" in data and data["all_pockets"]:
            valid = [p for p in data["all_pockets"] if p.get("boundary")]
            if valid:
                return valid

        if "best_pocket" in data and data["best_pocket"]:
            p = data["best_pocket"]
            if p.get("boundary"):
                return [p]

        return []
    except Exception:
        return []


def load_pocket_boundary_atoms(target_name: str, pocket_id):
    """加载指定口袋的边界原子坐标"""
    try:
        pocket_id_int = int(pocket_id)
    except (ValueError, TypeError):
        return None

    dirs = config.get_target_dirs(target_name)

    possible_paths = [
        dirs["pocket"] / f"pocket{pocket_id_int}_atm.pdb",
        dirs["pocket"] / f"fpocket_work/cleaned_out/pockets/pocket{pocket_id_int}_atm.pdb",
    ]

    pocket_pdb = None
    for path in possible_paths:
        if path.exists():
            pocket_pdb = path
            break

    if pocket_pdb is None:
        return None

    coords = []
    with open(pocket_pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except:
                    continue
    return np.array(coords) if coords else None


def get_ligand_atoms(ligand_pdbqt: Path, sample_ratio: float = 0.25) -> List[np.ndarray]:
    """从配体 PDBQT 中提取原子坐标"""
    coords = []
    in_first_model = False
    model_count = 0

    with open(ligand_pdbqt, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                model_count += 1
                if model_count == 1:
                    in_first_model = True
                else:
                    in_first_model = False
                continue
            if line.startswith('ENDMDL'):
                in_first_model = False
                continue
            if in_first_model and (line.startswith('ATOM') or line.startswith('HETATM')):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append(np.array([x, y, z]))
                except:
                    continue

    if not coords:
        return []

    n_sample = max(1, int(len(coords) * sample_ratio))
    return random.sample(coords, n_sample)


def is_inside_any_pocket(ligand_pdbqt: Path, pockets: List[Dict],
                         sample_ratio: float = 0.25,
                         target_name: str = None) -> Tuple[bool, str]:
    """检查配体是否在任意口袋内"""
    if not pockets:
        return True, "无口袋边界信息，跳过验证"

    atoms = get_ligand_atoms(ligand_pdbqt, sample_ratio)
    if not atoms:
        return False, "无法读取配体原子"

    # 尝试加载边界原子
    from scipy.spatial import KDTree
    pocket_trees = {}

    for pocket in pockets:
        pocket_id = pocket.get('id')
        if pocket_id is None:
            continue
        coords = load_pocket_boundary_atoms(target_name, pocket_id)
        if coords is not None and len(coords) > 0:
            pocket_trees[pocket_id] = KDTree(coords)

    if not pocket_trees:
        # 回退到边界盒子验证
        inside_count = 0
        for atom in atoms:
            for pocket in pockets:
                boundary = pocket.get('boundary')
                if boundary:
                    margin = config.VINA_VALIDATION.get("boundary_margin", 2.0)
                    if (boundary['x_min'] - margin <= atom[0] <= boundary['x_max'] + margin and
                        boundary['y_min'] - margin <= atom[1] <= boundary['y_max'] + margin and
                        boundary['z_min'] - margin <= atom[2] <= boundary['z_max'] + margin):
                        inside_count += 1
                        break
        ratio = inside_count / len(atoms) if atoms else 0
        threshold_ratio = config.VINA_VALIDATION.get("inside_threshold", 0.25)
        if ratio >= threshold_ratio:
            return True, f"{inside_count}/{len(atoms)} ({ratio*100:.0f}%) 原子在口袋内"
        else:
            return False, f"仅 {inside_count}/{len(atoms)} ({ratio*100:.0f}%) 原子在口袋内"

    threshold = config.VINA_VALIDATION.get("kd_tree_threshold", 3.5)
    atoms_inside = 0
    best_pocket_id = None

    for atom in atoms:
        for pocket_id, tree in pocket_trees.items():
            dist, _ = tree.query(atom)
            if dist < threshold:
                atoms_inside += 1
                if best_pocket_id is None:
                    best_pocket_id = pocket_id
                break

    threshold_ratio = config.VINA_VALIDATION.get("inside_threshold", 0.25)
    inside_ratio = atoms_inside / len(atoms) if atoms else 0

    if inside_ratio >= threshold_ratio:
        return True, f"{atoms_inside}/{len(atoms)} ({inside_ratio*100:.0f}%) 原子在口袋内"
    else:
        return False, f"仅 {atoms_inside}/{len(atoms)} ({inside_ratio*100:.0f}%) 原子在口袋内"


def validate_docking_result(ligand_pdbqt: Path,
                            binding_energy: float,
                            target_name: str = None,
                            pocket_center: Optional[np.ndarray] = None) -> Tuple[bool, str]:
    """验证对接结果"""
    from config import VINA_VALIDATION

    # 结合能范围检查
    min_energy = VINA_VALIDATION.get("min_energy", -15.0)
    max_energy = VINA_VALIDATION.get("max_energy", -3.0)

    if binding_energy > max_energy:
        return False, f"结合能太弱: {binding_energy:.2f} > {max_energy}"
    if binding_energy < min_energy:
        return False, f"结合能极端负值: {binding_energy:.2f} < {min_energy}"

    # 原子数检查
    atom_count = 0
    in_first_model = False
    model_count = 0

    with open(ligand_pdbqt, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                model_count += 1
                if model_count == 1:
                    in_first_model = True
                else:
                    in_first_model = False
                continue
            if line.startswith('ENDMDL'):
                in_first_model = False
                continue
            if in_first_model and (line.startswith('ATOM') or line.startswith('HETATM')):
                atom_count += 1

    min_atoms = VINA_VALIDATION.get("min_atoms", 30)
    max_atoms = VINA_VALIDATION.get("max_atoms", 2000)
    if atom_count < min_atoms or atom_count > max_atoms:
        return False, f"原子数 {atom_count} 不在范围 [{min_atoms}, {max_atoms}]"

    # 口袋边界验证
    use_pocket_boundary = VINA_VALIDATION.get("use_pocket_boundary", True)
    if use_pocket_boundary and target_name:
        pockets = load_pocket_boundaries(target_name)
        if pockets:
            sample_ratio = VINA_VALIDATION.get("sample_ratio", 0.25)
            inside, reason = is_inside_any_pocket(ligand_pdbqt, pockets, sample_ratio, target_name)
            if not inside:
                return False, f"口袋边界验证失败: {reason}"

    return True, "验证通过"


# =============================================================================
# GNINA 核心函数
# =============================================================================

def run_gnina(ligand_pdbqt: Path,
              receptor_pdb: Path,
              output_pdbqt: Path,
              target_name: str = None,
              pocket_center: Optional[List[float]] = None,
              box_size: Optional[List[float]] = None,
              timeout: int = 300,
              verbose: bool = False) -> Tuple[bool, Dict[str, float], str]:
    """
    调用 GNINA 进行对接（构象搜索）
    修复：输出写入临时日志文件，避免管道缓冲区死锁；解析使用读取文件得到的output_text，不再使用result.stdout
    """
    # 构建 GNINA 命令
    cmd = [
        "gnina",
        "-r", str(receptor_pdb),
        "-l", str(ligand_pdbqt),
        "-o", str(output_pdbqt),
        "--center_x", str(pocket_center[0]),
        "--center_y", str(pocket_center[1]),
        "--center_z", str(pocket_center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--exhaustiveness", "5",
        "--num_modes", "3",
        "--cnn_scoring", "rescore",
        # "--no_gpu"
    ]
    if verbose:
        print(f"  GNINA 命令: {' '.join(cmd)}")
    import tempfile
    import os
    # 创建临时日志文件，stdout/stderr重定向到文件，彻底避开PIPE
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as tmp_f:
        tmp_log_path = tmp_f.name
    try:
        with open(tmp_log_path, "w") as f_out:
            result = subprocess.run(
                cmd,
                stdout=f_out,
                stderr=f_out,
                text=True,
                timeout=timeout
            )
        # 读取完整输出（全部内容来自磁盘文件）
        with open(tmp_log_path, "r", encoding="utf-8") as f_log:
            output_text = f_log.read()

        if result.returncode != 0:
            return False, {}, f"GNINA 返回错误码 {result.returncode}, log:\n{output_text}"

        if verbose:
            print(f"【GNINA 调试】输出内容:")
            print(output_text)

        scores = {}
        best_affinity = None
        best_cnnscore = None
        best_cnnaffinity = None
        in_table = False

        # 【修复】全部迭代 output_text，不再使用 result.stdout
        for line in output_text.split('\n'):
            raw_line = line
            line = line.strip()
            # 宽松识别表头
            if "mode |" in line and "affinity" in line:
                in_table = True
                continue
            if '-----+' in line:
                continue
            if not in_table:
                continue
            if not line:
                continue
            parts = line.split()
            # 分割之后第一个token必须是数字mode编号
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    cnnscore = float(parts[2])
                    cnnaffinity = float(parts[3])
                    # 取第一个mode作为最优
                    if best_affinity is None:
                        best_affinity = affinity
                        best_cnnscore = cnnscore
                        best_cnnaffinity = cnnaffinity
                except (ValueError, IndexError):
                    continue

        # 只有主解析完全失败才进入备选解析
        if best_affinity is None:
            # 备选解析，增加防护，杜绝None.split()
            for line in output_text.split('\n'):
                if not line:
                    continue
                line = line.strip()
                try:
                    if 'Affinity:' in line:
                        _, val = line.split(":", 1)
                        best_affinity = float(val.strip())
                    elif 'CNNscore:' in line:
                        _, val = line.split(":", 1)
                        best_cnnscore = float(val.strip())
                    elif 'CNNaffinity:' in line:
                        _, val = line.split(":", 1)
                        best_cnnaffinity = float(val.strip())
                except (ValueError, IndexError):
                    continue

        # 最终校验
        if best_affinity is not None:
            scores['affinity'] = best_affinity
            if best_cnnscore is not None:
                scores['cnnscore'] = best_cnnscore
            if best_cnnaffinity is not None:
                scores['cnnaffinity'] = best_cnnaffinity
            if verbose:
                print(f"  【解析结果】Affinity: {scores.get('affinity', 0.0):.4f}")
                print(f"              CNNscore: {scores.get('cnnscore', 0.0):.4f}")
                print(f"              CNNaffinity: {scores.get('cnnaffinity', 0.0):.4f}")
            return True, scores, ""
        else:
            # 解析失败时，把完整日志放进错误消息，方便调试
            err_msg = f"GNINA解析失败，未提取到affinity数值。GNINA完整输出:\n{output_text}"
            return False, {}, err_msg

    except subprocess.TimeoutExpired:
        return False, {}, f"GNINA 超时（{timeout}秒）"
    except FileNotFoundError:
        return False, {}, "GNINA 未找到，请确保 gnina 在 PATH 中"
    except Exception as e:
        return False, {}, f"GNINA 执行异常: {e}"
    finally:
        # 清理临时日志
        if os.path.exists(tmp_log_path):
            os.unlink(tmp_log_path)


# =============================================================================
# 主要接口函数
# =============================================================================

def get_vina_paths(target_name: str) -> Dict[str, Path]:
    """获取文件路径"""
    dirs = config.get_target_dirs(target_name)
    receptor_pdb = dirs["cleaned"] / "cleaned.pdb"

    if not receptor_pdb.exists():
        raise FileNotFoundError(f"受体文件不存在: {receptor_pdb}")

    return {
        'receptor': receptor_pdb,
        'config': dirs["vina"] / "vina_config.txt"
    }


def run_vina_with_progress(ligand_pdbqt: Path,
                           receptor_pdbqt: Path,
                           vina_config: Path,
                           output_pdbqt: Optional[Path] = None,
                           timeout: int = 300,
                           n_cpu: Optional[int] = None,
                           exhaustiveness: Optional[int] = None,
                           num_modes: Optional[int] = None,
                           energy_range: Optional[int] = None,
                           verbose: bool = True,
                           sequence: str = "",
                           pocket_center: Optional[np.ndarray] = None,
                           validate_docking: bool = None,
                           target_name: str = None,
                           multi_pocket: bool = False) -> VinaResult:
    """
    运行 GNINA 打分（替代 Vina）
    """
    # 从vina_config.txt读取口袋中心与box尺寸
    # 从vina_config.txt读取口袋中心与box尺寸
    import re
    pocket_center = None
    box_size = None
    cx = cy = cz = None
    sx = sy = sz = None

    print(f"[DEBUG vina_config] vina_config path = {vina_config}")
    print(f"[DEBUG vina_config] exists? {vina_config.exists()}")

    if vina_config.exists():
        with open(vina_config, "r") as f:
            all_lines = f.readlines()
        print("[DEBUG vina_config] === file raw content START ===")
        for raw_line in all_lines:
            print(repr(raw_line))
        print("[DEBUG vina_config] === file raw content END ===")

        for line in all_lines:
            orig_line = line
            line = line.strip()
            if not line or line.startswith("#"):
                print(f"[DEBUG skip comment/empty] {repr(orig_line)}")
                continue

            # 支持两种写法： center_x 1.0   /   center_x = 1.0
            m_center = re.match(r"center_x\s*=\s*([-\d.]+)", line)
            if m_center:
                cx = float(m_center.group(1))
                print(f"[DEBUG match] center_x -> {cx}")

            m_center_y = re.match(r"center_y\s*=\s*([-\d.]+)", line)
            if m_center_y:
                cy = float(m_center_y.group(1))
                print(f"[DEBUG match] center_y -> {cy}")

            m_center_z = re.match(r"center_z\s*=\s*([-\d.]+)", line)
            if m_center_z:
                cz = float(m_center_z.group(1))
                print(f"[DEBUG match] center_z -> {cz}")

            m_sx = re.match(r"size_x\s*=\s*([-\d.]+)", line)
            if m_sx:
                sx = float(m_sx.group(1))
                print(f"[DEBUG match] size_x -> {sx}")

            m_sy = re.match(r"size_y\s*=\s*([-\d.]+)", line)
            if m_sy:
                sy = float(m_sy.group(1))
                print(f"[DEBUG match] size_y -> {sy}")

            m_sz = re.match(r"size_z\s*=\s*([-\d.]+)", line)
            if m_sz:
                sz = float(m_sz.group(1))
                print(f"[DEBUG match] size_z -> {sz}")

    # 打印全部解析出来的变量
    print(f"[DEBUG parsed vars] cx={cx}, cy={cy}, cz={cz} | sx={sx}, sy={sy}, sz={sz}")

    if cx is not None and cy is not None and cz is not None and sx is not None and sy is not None and sz is not None:
        pocket_center = [cx, cy, cz]
        box_size = [sx, sy, sz]
        print(f"[DEBUG SUCCESS] pocket_center={pocket_center}, box_size={box_size}")
    else:
        raise RuntimeError(f"vina_config.txt 无法解析得到center/size。\n"
                           f"解析结果: cx={cx}, cy={cy}, cz={cz}, sx={sx}, sy={sy}, sz={sz}\n"
                           f"请检查文件是否包含 center_x / center_y / center_z / size_x / size_y / size_z")


    if validate_docking is None:
        validate_docking = VINA_VALIDATION.get("enable", True)

    ligand_pdbqt = Path(ligand_pdbqt)
    receptor_pdbqt = Path(receptor_pdbqt)

    if not ligand_pdbqt.exists():
        raise FileNotFoundError(f"配体文件不存在: {ligand_pdbqt}")

    if not receptor_pdbqt.exists():
        raise FileNotFoundError(f"受体文件不存在: {receptor_pdbqt}")

    if output_pdbqt is None:
        output_pdbqt = ligand_pdbqt.parent / f"{ligand_pdbqt.stem}_gnina_out.pdbqt"
    else:
        output_pdbqt = Path(output_pdbqt)

    # ============================================================
    # 多口袋模式
    # ============================================================
    if multi_pocket and target_name:
        pockets = load_pocket_boundaries(target_name)
        if pockets and len(pockets) > 1:
            print(f"【多中心对接】使用 {len(pockets)} 个口袋分别对接")
            best_result = None
            best_energy = float('inf')

            for pocket in pockets:
                pocket_id = pocket.get('id', 'unknown')
                boundary = pocket.get('boundary')

                if not boundary:
                    continue

                pocket_center = boundary.get('center')
                if not pocket_center:
                    continue

                # 对该口袋运行 GNINA
                success, scores, error = run_gnina(
                    ligand_pdbqt=ligand_pdbqt,
                    receptor_pdb=receptor_pdbqt,
                    timeout=timeout,
                    verbose=verbose
                )

                if success:
                    energy = scores.get('affinity', 0.0)
                    if energy < 0 and energy < best_energy:
                        best_energy = energy
                        best_result = VinaResult(
                            binding_energy=energy,
                            output_file=output_pdbqt,
                            success=True,
                            sequence=sequence,
                            cnnscore=scores.get('cnnscore', 0.0),
                            cnnaffinity=scores.get('cnnaffinity', 0.0),
                            cnnvariance=scores.get('cnnvariance', 0.0)
                        )

            if best_result:
                return best_result

    # ============================================================
    # 单口袋模式
    # ============================================================
    success, scores, error_msg = run_gnina(
        ligand_pdbqt=ligand_pdbqt,
        receptor_pdb=receptor_pdbqt,
        output_pdbqt=output_pdbqt,
        target_name=target_name,
        pocket_center=pocket_center,
        box_size=box_size,
        timeout=timeout,
        verbose=verbose
    )

    if not success:
        return VinaResult(
            binding_energy=0.0,
            output_file=output_pdbqt,
            success=False,
            error_message=error_msg,
            sequence=sequence,
            validation_passed=False
        )

    binding_energy = scores.get('affinity', 0.0)

    if verbose:
        seq_info = f"[{sequence}] " if sequence else ""
        print(f"{seq_info}GNINA 打分完成:")
        print(f"  Affinity: {binding_energy:.4f} kcal/mol")
        print(f"  CNNscore: {scores.get('cnnscore', 0.0):.4f}")
        print(f"  CNNaffinity: {scores.get('cnnaffinity', 0.0):.4f}")
        print(f"  CNNvariance: {scores.get('cnnvariance', 0.0):.4f}")

    # 验证
    # 验证（如果启用）
    validation_passed = True
    validation_reason = "验证通过"
    if validate_docking:
        # 【修复】GNINA 的 --score_only 模式不生成输出文件
        # 使用原始配体文件进行验证，而不是 output_pdbqt
        validation_file = output_pdbqt if output_pdbqt.exists() else ligand_pdbqt

        if validation_file.exists():
            is_valid, reason = validate_docking_result(
                ligand_pdbqt=validation_file,
                binding_energy=binding_energy,
                target_name=target_name,
                pocket_center=pocket_center
            )
            validation_passed = is_valid
            validation_reason = reason
            if verbose:
                if is_valid:
                    print(f"  ✓ 验证通过")
                else:
                    print(f"  ✗ 验证失败: {reason}")
        else:
            # 没有可用的文件进行验证，跳过
            if verbose:
                print(f"  ⚠ 无验证文件，跳过验证")

    return VinaResult(
        binding_energy=binding_energy,
        output_file=output_pdbqt,
        success=True,
        error_message="",
        sequence=sequence,
        validation_passed=validation_passed,
        validation_reason=validation_reason,
        cnnscore=scores.get('cnnscore', 0.0),
        cnnaffinity=scores.get('cnnaffinity', 0.0),
        cnnvariance=scores.get('cnnvariance', 0.0)
    )


def vina_dock(sequence: str,
              target_name: str,
              crosslinker: Optional[str] = None,
              crosslinker_positions: Optional[list] = None,
              output_dir: Optional[Path] = None,
              timeout: int = 300,
              n_cpu: Optional[int] = None,
              exhaustiveness: Optional[int] = None,
              verbose: bool = False,
              validate_docking: bool = None) -> float:
    """
    主函数：序列 → GNINA 结合能
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"GNINA 对接: {sequence}")
        print(f"靶点: {target_name}")
        print(f"{'='*60}")

    if crosslinker is None:
        crosslinker = config.CROSSLINKER
    if crosslinker_positions is None:
        crosslinker_positions = config.CROSSLINKER_POSITIONS
    if validate_docking is None:
        validate_docking = VINA_VALIDATION.get("enable", True)

    try:
        vina_paths = get_vina_paths(target_name)

        if verbose:
            print(f"\n[1/2] 生成分子...")

        pdbqt_path = generate_ligand(
            sequence=sequence,
            crosslinker=crosslinker,
            crosslinker_positions=crosslinker_positions,
            output_dir=output_dir
        )

        if verbose:
            print(f"  ✓ PDBQT: {pdbqt_path}")

        if verbose:
            print(f"\n[2/2] GNINA 打分...")

        result = run_vina_with_progress(
            ligand_pdbqt=pdbqt_path,
            receptor_pdbqt=vina_paths['receptor'],
            vina_config=vina_paths['config'],
            timeout=timeout,
            verbose=verbose,
            sequence=sequence,
            validate_docking=validate_docking,
            target_name=target_name
        )

        if not result.success:
            if verbose:
                print(f"  ✗ 失败: {result.error_message}")
            return 0.0

        if verbose:
            print(f"{'='*60}\n")

        return result.binding_energy

    except Exception as e:
        if verbose:
            print(f"  ✗ 异常: {e}")
        return 0.0


def dock_single_worker(args):
    sequence, target_name, crosslinker, crosslinker_positions, output_dir, timeout, n_cpu, exhaustiveness, validate_docking = args
    energy = vina_dock(
        sequence=sequence,
        target_name=target_name,
        crosslinker=crosslinker,
        crosslinker_positions=crosslinker_positions,
        output_dir=output_dir,
        timeout=timeout,
        n_cpu=n_cpu,
        exhaustiveness=exhaustiveness,
        verbose=True,
        validate_docking=validate_docking
    )
    return sequence, energy


def batch_vina_dock(sequences: List[str],
                    target_name: str,
                    output_file: Optional[Path] = None,
                    parallel: bool = False,
                    n_workers: Optional[int] = None,
                    n_cpu_per_worker: Optional[int] = None,
                    timeout: int = 300,
                    verbose: bool = False,
                    validate_docking: bool = None) -> Dict[str, float]:
    """批量对接"""
    import multiprocessing

    if validate_docking is None:
        validate_docking = VINA_VALIDATION.get("enable", True)

    if parallel:
        if n_workers is None:
            n_workers = config.PARALLEL_VINA_CONFIG.get("num_workers", multiprocessing.cpu_count())
        if n_cpu_per_worker is None:
            n_cpu_per_worker = config.VINA_CONFIG.get("cpu", 4)

        print(f"批量 GNINA 对接（并行模式）")
        print(f"总序列数: {len(sequences)}")
        print(f"并行进程数: {n_workers}")
        print("="*60)

        crosslinker = config.CROSSLINKER
        crosslinker_positions = config.CROSSLINKER_POSITIONS
        exhaustiveness = config.VINA_CONFIG.get("exhaustiveness", 4)
        output_dir = config.BASE_DIR / "temp" / "vina_dock"
        output_dir.mkdir(parents=True, exist_ok=True)

        args_list = [
            (seq, target_name, crosslinker, crosslinker_positions, output_dir, timeout, n_cpu_per_worker, exhaustiveness, validate_docking)
            for seq in sequences
        ]

        results = {}
        completed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(dock_single_worker, args): args[0] for args in args_list}

            for future in as_completed(futures):
                sequence = futures[future]
                try:
                    seq, energy = future.result()
                    results[seq] = energy
                    completed += 1
                    if energy == 0:
                        failed += 1
                except Exception as e:
                    results[sequence] = 0.0
                    completed += 1
                    failed += 1
                    print(f"✗ [{sequence}] 异常: {e}")

                progress = completed / len(sequences) * 100
                print(f"\n>>> 总进度: {completed}/{len(sequences)} ({progress:.1f}%) | 成功: {completed-failed} | 失败: {failed}\n")

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write("sequence,energy\n")
                for seq, energy in results.items():
                    f.write(f"{seq},{energy}\n")

        return results

    else:
        results = {}
        for i, seq in enumerate(sequences):
            if verbose:
                print(f"\n[{i+1}/{len(sequences)}] 处理序列: {seq}")
            energy = vina_dock(seq, target_name, verbose=verbose, validate_docking=validate_docking)
            results[seq] = energy

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write("sequence,energy\n")
                for seq, energy in results.items():
                    f.write(f"{seq},{energy}\n")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='GNINA 对接集成')
    parser.add_argument('-s', '--sequence', type=str, default=None, help='氨基酸序列')
    parser.add_argument('-l', '--list', type=str, default=None, help='序列列表文件')
    parser.add_argument('-t', '--target', type=str, required=True, help='靶点名称')
    parser.add_argument('-o', '--output', type=Path, default=None, help='输出文件')
    parser.add_argument('--timeout', type=int, default=300, help='超时时间（秒）')
    parser.add_argument('--parallel', action='store_true', help='使用并行模式')
    parser.add_argument('--workers', type=int, default=None, help='并行工作进程数')
    parser.add_argument('--no-validate', action='store_true', help='禁用对接结果验证')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    validate_docking = not args.no_validate

    if args.sequence:
        energy = vina_dock(
            sequence=args.sequence,
            target_name=args.target,
            timeout=args.timeout,
            verbose=args.verbose,
            validate_docking=validate_docking
        )
        print(f"\n结合能: {energy:.4f} kcal/mol")

    elif args.list:
        sequences = []
        with open(args.list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and ',' not in line:
                    sequences.append(line)

        results = batch_vina_dock(
            sequences=sequences,
            target_name=args.target,
            output_file=args.output,
            parallel=args.parallel,
            n_workers=args.workers,
            timeout=args.timeout,
            verbose=args.verbose,
            validate_docking=validate_docking
        )

        energies = [e for e in results.values() if e != 0]
        if energies:
            print(f"\n统计:")
            print(f"  成功: {len(energies)}/{len(sequences)}")
            print(f"  最佳结合能: {min(energies):.4f} kcal/mol")
            print(f"  平均结合能: {sum(energies)/len(energies):.4f} kcal/mol")

    else:
        print("错误: 请指定 -s（单序列）或 -l（序列列表文件）")


if __name__ == "__main__":
    main()