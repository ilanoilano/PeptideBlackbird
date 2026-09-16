#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vina 受体准备模块
功能：将 PDB 转换为 Vina 可识别的 PDBQT 格式，生成对接配置

修复：使用 OpenBabel 生成 PDBQT，不再手写
      【关键修复】使用 -xr 选项生成刚性受体，避免 ROOT/BRANCH 标签
      【修复】配置文件从 config.VINA_CONFIG 读取参数

输出：
  - results/[target_name]/vina/vina-receptor.pdbqt
  - results/[target_name]/vina/vina_config.txt
"""

import os
import sys
import random
import subprocess
import json
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

from config import get_target_dirs, TOOLS, VINA_BOX_SIZE, VINA_CONFIG


def pdb_to_pdbqt_obabel(input_pdb: Path, output_pdbqt: Path) -> bool:
    """
    使用 OpenBabel 将 PDB 转换为 PDBQT（刚性受体格式）

    修复：使用 -xr 选项生成刚性受体，避免 ROOT/BRANCH 等柔性标签

    Args:
        input_pdb: 输入 PDB 文件路径
        output_pdbqt: 输出 PDBQT 文件路径

    Returns:
        是否成功
    """
    input_pdb = Path(input_pdb)
    output_pdbqt = Path(output_pdbqt)

    if not input_pdb.exists():
        print(f"【错误】输入 PDB 文件不存在: {input_pdb}")
        return False

    # 获取 obabel 路径
    obabel_path = TOOLS.get("obabel", "obabel")

    print(f"使用 OpenBabel 生成刚性受体 PDBQT...")
    print(f"  输入: {input_pdb}")
    print(f"  输出: {output_pdbqt}")

    # 构建命令
    # -p: 计算 Gasteiger 电荷（部分电荷）
    # -xr: 生成刚性受体（去除 ROOT/BRANCH 等柔性标签，关键！）
    cmd = [
        obabel_path,
        str(input_pdb),
        "-opdbqt",
        "-p",  # 添加部分电荷
        "-xr",  # 【关键】刚性受体模式，去除柔性标签
        "-O", str(output_pdbqt)
    ]

    print(f"  命令: {' '.join(cmd)}")

    try:
        # 执行转换（不使用 shell=True，避免注入风险）
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"【错误】OpenBabel 转换失败:")
            print(f"  返回码: {result.returncode}")
            print(f"  错误输出: {result.stderr[:500]}")
            return False

        # 验证输出文件
        if not output_pdbqt.exists():
            print(f"【错误】输出文件未生成: {output_pdbqt}")
            return False

        if output_pdbqt.stat().st_size == 0:
            print(f"【错误】输出文件为空: {output_pdbqt}")
            return False

        # 检查 PDBQT 格式
        with open(output_pdbqt, 'r') as f:
            content = f.read()

        if 'ATOM' not in content and 'HETATM' not in content:
            print(f"【错误】PDBQT 文件缺少原子记录")
            return False

        # 【新增】验证是否包含柔性标签（不应该有）
        flexible_tags = ['ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF']
        found_tags = [tag for tag in flexible_tags if tag in content]
        if found_tags:
            print(f"【警告】受体文件包含柔性标签: {found_tags}")
            print(f"  这会导致 Vina 报错，尝试重新生成...")
            return False

        # 统计原子数
        atom_count = content.count('ATOM') + content.count('HETATM')
        print(f"✓ 刚性受体 PDBQT 生成成功: {output_pdbqt}")
        print(f"  原子数: {atom_count}")

        return True

    except subprocess.TimeoutExpired:
        print(f"【错误】OpenBabel 转换超时（60秒）")
        return False
    except FileNotFoundError:
        print(f"【错误】OpenBabel 未找到: {obabel_path}")
        print(f"  请确保 OpenBabel 已安装并在 PATH 中")
        print(f"  或检查 config.TOOLS['obabel'] 设置")
        return False
    except Exception as e:
        print(f"【错误】OpenBabel 执行异常: {e}")
        return False


def create_vina_config(pocket_center: Tuple[float, float, float],
                       box_size: Tuple[float, float, float],
                       output_config: Path,
                       receptor_pdbqt: Path = None,
                       target_name: str = None):
    """
    创建 Vina 对接配置文件

    【修改】如果提供了 target_name，从 pocket.json 读取所有口袋的全局边界
    生成覆盖所有口袋的大盒子

    Args:
        pocket_center: 口袋中心 (x, y, z)（当没有 target_name 时使用）
        box_size: 盒子尺寸 (x, y, z)（当没有 target_name 时使用）
        output_config: 输出配置文件路径
        receptor_pdbqt: 受体 PDBQT 路径（可选）
        target_name: 靶点名称（用于读取所有口袋边界）
    """
    # 【新增】如果提供了 target_name，从 pocket.json 读取所有口袋的全局边界
    if target_name is not None:
        pocket_json = Path(f"results/{target_name}/pocket/pocket.json")
        if pocket_json.exists():
            import json
            with open(pocket_json, 'r') as f:
                data = json.load(f)

            all_pockets = data.get("all_pockets", [])
            if all_pockets:
                # 计算所有口袋的全局边界
                x_min = min(p['boundary']['x_min'] for p in all_pockets if p.get('boundary'))
                x_max = max(p['boundary']['x_max'] for p in all_pockets if p.get('boundary'))
                y_min = min(p['boundary']['y_min'] for p in all_pockets if p.get('boundary'))
                y_max = max(p['boundary']['y_max'] for p in all_pockets if p.get('boundary'))
                z_min = min(p['boundary']['z_min'] for p in all_pockets if p.get('boundary'))
                z_max = max(p['boundary']['z_max'] for p in all_pockets if p.get('boundary'))

                # 计算全局中心
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                center_z = (z_min + z_max) / 2

                # 计算全局尺寸（+6Å 余量）
                margin = 6.0
                size_x = (x_max - x_min) + margin
                size_y = (y_max - y_min) + margin
                size_z = (z_max - z_min) + margin

                # 确保最小尺寸为 20Å
                size_x = max(size_x, 20.0)
                size_y = max(size_y, 20.0)
                size_z = max(size_z, 20.0)

                pocket_center = (center_x, center_y, center_z)
                box_size = (size_x, size_y, size_z)

                print(f"  【全局盒子】覆盖 {len(all_pockets)} 个口袋")
                print(f"    中心: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
                print(f"    尺寸: ({size_x:.1f}, {size_y:.1f}, {size_z:.1f})")

    # 【修复】从 config 读取 Vina 参数
    exhaustiveness = VINA_CONFIG.get("exhaustiveness", 4)
    num_modes = VINA_CONFIG.get("num_modes", 9)
    energy_range = VINA_CONFIG.get("energy_range", 4)
    cpu = VINA_CONFIG.get("cpu", 4)

    config_content = f"""# Vina 对接配置
# Auto-generated by MCTS-Peptide Pipeline

# 受体
{"receptor = " + str(receptor_pdbqt) if receptor_pdbqt else "# receptor = path/to/receptor.pdbqt"}

# 搜索空间（对接盒子）
center_x = {pocket_center[0]:.3f}
center_y = {pocket_center[1]:.3f}
center_z = {pocket_center[2]:.3f}

size_x = {box_size[0]:.1f}
size_y = {box_size[1]:.1f}
size_z = {box_size[2]:.1f}

# 对接参数（从 config.VINA_CONFIG 读取）
exhaustiveness = {exhaustiveness}
num_modes = {num_modes}
energy_range = {energy_range}
cpu = {cpu}
"""

    output_config = Path(output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)

    with open(output_config, 'w') as f:
        f.write(config_content)

    print(f"✓ Vina 配置: {output_config}")
    print(f"  参数: exhaustiveness={exhaustiveness}, cpu={cpu}")


def prepare_for_vina(target_name: str):
    """
    主入口：准备 Vina 受体文件

    Args:
        target_name: 靶点名称
    """
    print(f"\n{'=' * 60}")
    print(f"准备 Vina 受体: {target_name}")
    print(f"{'=' * 60}")

    dirs = get_target_dirs(target_name)

    # 输入文件
    cleaned_pdb = dirs["cleaned"] / "cleaned.pdb"
    pocket_json = dirs["pocket"] / "pocket.json"

    if not cleaned_pdb.exists():
        raise FileNotFoundError(f"请先运行 pdb_cleaner.py: {cleaned_pdb}")

    if not pocket_json.exists():
        raise FileNotFoundError(f"请先运行 pdb_to_pockets.py: {pocket_json}")

    # 读取口袋信息
    with open(pocket_json, 'r') as f:
        pocket_data = json.load(f)

    best_pocket = pocket_data.get("best_pocket", {})
    pocket_center = best_pocket.get("center")

    if not pocket_center:
        raise ValueError(f"无法读取口袋中心: {pocket_json}")

    print(f"  最佳口袋中心: ({pocket_center[0]:.3f}, {pocket_center[1]:.3f}, {pocket_center[2]:.3f})")

    # 1. 转换为 PDBQT（使用 OpenBabel，刚性受体模式）
    output_pdbqt = dirs["vina"] / "vina-receptor.pdbqt"
    success = pdb_to_pdbqt_obabel(cleaned_pdb, output_pdbqt)

    if not success:
        raise RuntimeError("PDB 转 PDBQT 失败")

    # 2. 创建 Vina 配置
    output_config = dirs["vina"] / "vina_config.txt"
    create_vina_config(
        pocket_center=tuple(pocket_center),
        box_size=VINA_BOX_SIZE,
        output_config=output_config,
        receptor_pdbqt=output_pdbqt,
        target_name=target_name  # 【新增】传入靶点名称
    )

    print(f"\n{'=' * 60}")
    print(f"✓ Vina 受体准备完成")
    print(f"{'=' * 60}")

    return output_pdbqt


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="准备 Vina 受体文件")
    parser.add_argument("target", help="靶点名称（如 1LYZ）")

    args = parser.parse_args()

    try:
        prepare_for_vina(args.target)
    except Exception as e:
        print(f"\n【错误】{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()