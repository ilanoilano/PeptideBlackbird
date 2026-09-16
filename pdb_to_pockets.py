# -*- coding: utf-8 -*-
"""
蛋白口袋检测模块
功能：调用 fpocket 识别结合口袋，提取中心坐标和残基列表
输出：
  - results/[target_name]/pocket/pocket.json
  - results/[target_name]/pocket/pocket-for-esmif.pdb
  - results/[target_name]/pocket/pocket_seq.txt
"""

# -*- coding: utf-8 -*-
"""
蛋白口袋检测模块
功能：调用 fpocket 识别结合口袋，提取中心坐标和残基列表
【修改】解析所有口袋的边界原子，计算边界盒子

输出：
  - results/[target_name]/pocket/pocket.json
  - results/[target_name]/pocket/pocket-for-esmif.pdb
  - results/[target_name]/pocket/pocket_seq.txt
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import get_target_dirs, TOOLS


def run_fpocket(cleaned_pdb: Path, output_dir: Path) -> Path:
    """
    运行 fpocket 检测口袋

    Args:
        cleaned_pdb: 清洗后的 PDB 文件路径
        output_dir: 输出目录

    Returns:
        fpocket 输出目录路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # fpocket 会在输入文件所在目录创建输出目录
    # 我们需要先复制到工作目录
    work_dir = output_dir / "fpocket_work"
    work_dir.mkdir(exist_ok=True)

    import shutil
    work_pdb = work_dir / cleaned_pdb.name
    shutil.copy2(cleaned_pdb, work_pdb)

    print(f"  运行 fpocket...")
    cmd = [TOOLS["fpocket"], "-f", str(work_pdb)]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"  fpocket 错误: {e.stderr}")
        raise RuntimeError("fpocket 执行失败")
    except FileNotFoundError:
        print("【需要安装】fpocket")
        print("请运行: conda install -c conda-forge fpocket 或从源码编译")
        raise RuntimeError("fpocket 未找到")

    # fpocket 输出目录: {pdb_name}_out
    fpocket_out = work_dir / f"{work_pdb.stem}_out"

    if not fpocket_out.exists():
        raise RuntimeError(f"fpocket 输出目录不存在: {fpocket_out}")

    print(f"  ✓ fpocket 完成: {fpocket_out}")
    return fpocket_out


def parse_pocket_boundary(pocket_pdb: Path) -> Dict[str, float]:
    """
    从 pocketX_atm.pdb 解析边界盒子

    读取所有原子坐标，计算 min/max 范围

    Args:
        pocket_pdb: pocketX_atm.pdb 文件路径

    Returns:
        {
            'x_min': float, 'x_max': float,
            'y_min': float, 'y_max': float,
            'z_min': float, 'z_max': float,
            'atom_count': int
        }
    """
    coords = []

    if not pocket_pdb.exists():
        return {
            'x_min': 0, 'x_max': 0,
            'y_min': 0, 'y_max': 0,
            'z_min': 0, 'z_max': 0,
            'atom_count': 0
        }

    with open(pocket_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue

    if not coords:
        return {
            'x_min': 0, 'x_max': 0,
            'y_min': 0, 'y_max': 0,
            'z_min': 0, 'z_max': 0,
            'atom_count': 0
        }

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    return {
        'x_min': min(xs),
        'x_max': max(xs),
        'y_min': min(ys),
        'y_max': max(ys),
        'z_min': min(zs),
        'z_max': max(zs),
        'atom_count': len(coords),
        'center': (
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            (min(zs) + max(zs)) / 2
        )
    }


def parse_fpocket_results(fpocket_out: Path, pdb_stem: str) -> List[Dict]:
    """
    解析 fpocket 输出结果

    【修改】新增解析 pocketX_atm.pdb 获取边界盒子

    Args:
        fpocket_out: fpocket 输出目录
        pdb_stem: PDB 文件名（不含扩展名）

    Returns:
        口袋列表，每个口袋包含：
        - id: 口袋编号
        - score: fpocket 评分
        - volume: 体积（Å³）
        - center: 中心坐标 (x, y, z)
        - residues: 口袋残基列表 [(chain, res_id, res_name), ...]
        - boundary: 边界盒子 {x_min, x_max, y_min, y_max, z_min, z_max, atom_count, center}
    """
    pockets = []

    # 读取 info.txt 获取评分和体积（fpocket 4.0 格式: {pdb_name}_info.txt）
    info_file = fpocket_out / f"{pdb_stem}_info.txt"
    if not info_file.exists():
        # 尝试旧格式
        info_file = fpocket_out / "info.txt"

    if info_file.exists():
        with open(info_file, 'r') as f:
            content = f.read()

        # 解析多行格式
        current_pocket = None
        for line in content.split('\n'):
            line = line.strip()

            # 新口袋开始: "Pocket 1 :"
            if line.startswith("Pocket") and ':' in line:
                parts = line.split()
                pocket_id = int(parts[1])
                current_pocket = {
                    "id": pocket_id,
                    "score": None,
                    "volume": None,
                    "center": None,
                    "residues": [],
                    "boundary": None
                }
                pockets.append(current_pocket)

            # 解析 Score
            elif line.startswith("Score :") and current_pocket is not None:
                try:
                    current_pocket["score"] = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass

            # 解析 Volume
            elif line.startswith("Volume :") and current_pocket is not None:
                try:
                    current_pocket["volume"] = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass

    # 读取每个口袋的 PDB 文件获取残基信息和边界盒子
    pockets_dir = fpocket_out / "pockets"
    for pocket in pockets:
        # 尝试新格式（pockets/ 子目录）
        pocket_pdb = pockets_dir / f"pocket{pocket['id']}_atm.pdb"
        if not pocket_pdb.exists():
            # 尝试旧格式（根目录）
            pocket_pdb = fpocket_out / f"pocket{pocket['id']}_atm.pdb"

        if not pocket_pdb.exists():
            print(f"    警告: 口袋 {pocket['id']} 的 PDB 文件不存在: {pocket_pdb}")
            continue

        # 【新增】解析边界盒子
        boundary = parse_pocket_boundary(pocket_pdb)
        pocket["boundary"] = boundary

        residues = set()
        atoms = []

        with open(pocket_pdb, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # 解析 ATOM 记录
                    chain_id = line[21].strip()
                    res_seq = line[22:26].strip()
                    res_name = line[17:20].strip()

                    residues.add((chain_id, res_seq, res_name))

                    # 解析坐标
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        atoms.append((x, y, z))
                    except ValueError:
                        continue

        # 计算几何中心（如果边界中没有center，使用原子中心）
        if atoms:
            center_x = sum(a[0] for a in atoms) / len(atoms)
            center_y = sum(a[1] for a in atoms) / len(atoms)
            center_z = sum(a[2] for a in atoms) / len(atoms)
            pocket["center"] = (round(center_x, 3), round(center_y, 3), round(center_z, 3))

        pocket["residues"] = sorted(list(residues))

    # 按评分排序（分数越低越好）
    pockets.sort(key=lambda x: x.get("score", float('inf')))

    return pockets


def select_best_pocket(pockets: List[Dict], top_n: int = 1) -> List[Dict]:
    """
    选择最佳口袋

    策略：
    1. 优先选择 fpocket 评分最低的（最可成药）
    2. 体积适中（100-1000 Å³）
    """
    # 过滤掉没有 score 或 volume 的口袋
    valid_pockets = [
        p for p in pockets
        if p.get("score") is not None and p.get("volume") is not None
           and 100 <= p["volume"] <= 1000
    ]

    if not valid_pockets:
        # 如果没有体积合格的，返回有 score 的
        valid_pockets = [p for p in pockets if p.get("score") is not None]

    if not valid_pockets:
        valid_pockets = pockets

    # 按评分排序
    valid_pockets.sort(key=lambda x: x.get("score", float('inf')))

    return valid_pockets[:top_n]


def create_pocket_pdb(cleaned_pdb: Path, pocket: Dict, output_pdb: Path):
    """
    创建包含口袋残基的 PDB 文件（用于 ESM-IF）

    提取口袋残基对应的所有原子
    """
    pocket_residues = set((r[0], r[1]) for r in pocket["residues"])  # (chain, res_id)

    with open(cleaned_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                res_seq = line[22:26].strip()

                if (chain_id, res_seq) in pocket_residues:
                    f_out.write(line)
            elif line.startswith("TER") or line.startswith("END"):
                f_out.write(line)

    print(f"  ✓ 口袋 PDB: {output_pdb}")


def create_pocket_sequence(pocket: Dict, output_txt: Path):
    """
    创建口袋残基序列文件
    """
    # 按残基编号排序
    residues = sorted(pocket["residues"], key=lambda x: (x[0], int(x[1])))

    # 三字母转单字母
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    with open(output_txt, 'w') as f:
        f.write(f">Pocket_{pocket['id']} Score={pocket['score']:.2f} Volume={pocket['volume']:.1f}\n")

        seq = ""
        for chain, res_id, res_name in residues:
            aa = three_to_one.get(res_name, "X")
            seq += aa
            f.write(f"{chain}:{res_id} {res_name} {aa}\n")

        f.write(f"\nSequence: {seq}\n")

    print(f"  ✓ 口袋序列: {output_txt}")


def detect_pockets(target_name: str, top_n: int = 1) -> Dict:
    """
    主入口：检测蛋白口袋

    【修改】保存所有口袋到 pocket.json，包含边界信息

    Args:
        target_name: 靶点名称
        top_n: 返回前 N 个口袋（用于 ESM-IF）

    Returns:
        最佳口袋信息字典
    """
    print(f"\n{'=' * 60}")
    print(f"检测口袋: {target_name}")
    print(f"{'=' * 60}")

    dirs = get_target_dirs(target_name)
    cleaned_pdb = dirs["cleaned"] / "cleaned.pdb"

    if not cleaned_pdb.exists():
        raise FileNotFoundError(f"请先运行 pdb_cleaner.py: {cleaned_pdb}")

    # 1. 运行 fpocket
    fpocket_out = run_fpocket(cleaned_pdb, dirs["pocket"])

    # 2. 解析结果（包含所有口袋的边界盒子）
    print(f"  解析 fpocket 结果...")
    all_pockets = parse_fpocket_results(fpocket_out, cleaned_pdb.stem)
    print(f"  发现 {len(all_pockets)} 个口袋")

    # 3. 选择最佳口袋（用于 ESM-IF）
    best_pockets = select_best_pocket(all_pockets, top_n)

    if not best_pockets:
        raise RuntimeError("未找到合适的口袋")

    best_pocket = best_pockets[0]
    print(f"\n  最佳口袋 #{best_pocket['id']}:")
    print(f"    评分: {best_pocket['score']:.2f}" if best_pocket.get('score') else "    评分: N/A")
    print(f"    体积: {best_pocket['volume']:.1f} Å³" if best_pocket.get('volume') else "    体积: N/A")
    if best_pocket.get('center'):
        print(
            f"    中心: ({best_pocket['center'][0]:.3f}, {best_pocket['center'][1]:.3f}, {best_pocket['center'][2]:.3f})")
    else:
        print(f"    中心: N/A")
    print(f"    残基数: {len(best_pocket.get('residues', []))}")

    # 【新增】统计所有口袋的边界信息
    pocket_dir = dirs["pocket"]

    # 4. 保存所有口袋信息到 pocket.json（包含边界盒子）
    pocket_json = pocket_dir / "pocket.json"

    # 准备所有口袋数据（去除过大的边界数据，只保留必要的）
    all_pockets_serializable = []
    for p in all_pockets:
        p_copy = p.copy()
        if p_copy.get('boundary'):
            # 保留边界信息（用于 Vina 验证）
            p_copy['boundary'] = p_copy['boundary']
        all_pockets_serializable.append(p_copy)

    with open(pocket_json, 'w') as f:
        json.dump({
            "target": target_name,
            "best_pocket": best_pocket,  # 保留最佳口袋（兼容性）
            "all_pockets": all_pockets_serializable,  # 所有口袋（新增）
            "total_pockets": len(all_pockets)  # 口袋总数（新增）
        }, f, indent=2, default=str)  # default=str 处理可能的非序列化类型
    print(f"  ✓ 口袋 JSON: {pocket_json} (包含 {len(all_pockets)} 个口袋)")

    # 5. pocket-for-esmif.pdb（仅最佳口袋）
    pocket_pdb = pocket_dir / "pocket-for-esmif.pdb"
    original_cleaned = dirs["cleaned"] / "cleaned.pdb"
    create_pocket_pdb(original_cleaned, best_pocket, pocket_pdb)

    # 6. pocket_seq.txt（仅最佳口袋）
    pocket_seq = pocket_dir / "pocket_seq.txt"
    create_pocket_sequence(best_pocket, pocket_seq)

    return best_pocket


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="检测蛋白结合口袋")
    parser.add_argument("target", help="靶点名称")
    parser.add_argument("--top-n", type=int, default=1, help="返回前 N 个口袋")

    args = parser.parse_args()

    try:
        pocket = detect_pockets(args.target, args.top_n)
        print(f"\n✓ 口袋检测完成")
        print(f"  中心坐标: ({pocket['center'][0]:.3f}, {pocket['center'][1]:.3f}, {pocket['center'][2]:.3f})")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()