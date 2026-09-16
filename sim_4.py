#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim_4: Vina 分子对接模块
功能：运行 AutoDock Vina 分子对接

输入：
- 配体 PDBQT 文件
- 受体 PDBQT 文件（从 config 读取路径）
- Vina 配置（从 config 读取或命令行指定）

输出：
- 对接结果 PDBQT 文件
- 结合能数据
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import config

# 输出目录
OUTPUT_DIR = Path("/mnt/d/code/AA/temporary")


@dataclass
class VinaResult:
    """Vina 对接结果"""
    rank: int
    binding_energy: float  # kcal/mol
    rmsd_lb: float         # RMSD lower bound
    rmsd_ub: float         # RMSD upper bound


@dataclass
class VinaOutput:
    """Vina 完整输出"""
    results: List[VinaResult]
    output_file: Path
    best_energy: float
    
    def get_best(self) -> VinaResult:
        """获取最佳结果"""
        return self.results[0] if self.results else None


def get_vina_config(target_name: str) -> Optional[Path]:
    """获取 Vina 配置文件路径"""
    dirs = config.get_target_dirs(target_name)
    vina_config = dirs["vina"] / "vina_config.txt"
    return vina_config if vina_config.exists() else None


def get_receptor_pdbqt(target_name: str) -> Optional[Path]:
    """获取受体 PDBQT 文件路径"""
    dirs = config.get_target_dirs(target_name)
    receptor_pdbqt = dirs["vina"] / "vina-receptor.pdbqt"
    return receptor_pdbqt if receptor_pdbqt.exists() else None


def parse_vina_output(stdout: str) -> List[VinaResult]:
    """
    解析 Vina 输出，提取结合能信息
    
    Args:
        stdout: Vina 标准输出
    
    Returns:
        结合能结果列表
    """
    results = []
    lines = stdout.split('\n')
    in_table = False
    
    for line in lines:
        line = line.strip()
        
        # 检测表格开始
        if '-----+------------+----------+----------' in line:
            in_table = True
            continue
        
        # 解析表格行
        if in_table and line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    rank = int(parts[0])
                    binding_energy = float(parts[1])
                    rmsd_lb = float(parts[2])
                    rmsd_ub = float(parts[3])
                    
                    results.append(VinaResult(
                        rank=rank,
                        binding_energy=binding_energy,
                        rmsd_lb=rmsd_lb,
                        rmsd_ub=rmsd_ub
                    ))
                except ValueError:
                    continue
    
    return results


def run_vina_docking(
    ligand_pdbqt: Path,
    target_name: str = None,
    receptor_pdbqt: Path = None,
    vina_config: Path = None,
    output_file: Path = None,
    exhaustiveness: int = None,
    num_modes: int = None,
    timeout: int = 300,
    verbose: bool = True
) -> VinaOutput:
    """
    运行 Vina 分子对接（Popen版本，处理容器OpenMP死锁，超时kill进程）
    Args:
        ligand_pdbqt: 配体 PDBQT 文件路径
        target_name: 靶点名称（用于自动查找受体和配置）
        receptor_pdbqt: 受体 PDBQT 文件路径（优先于 target_name）
        vina_config: Vina 配置文件路径（优先于 target_name）
        output_file: 输出文件路径（默认自动生成）
        exhaustiveness: 搜索详尽度（覆盖配置文件）
        num_modes: 输出构象数量（覆盖配置文件）
        timeout: 超时时间（秒）
        verbose: 是否打印详细信息
    Returns:
        VinaOutput 对象
    """
    ligand_pdbqt = Path(ligand_pdbqt)

    if not ligand_pdbqt.exists():
        raise FileNotFoundError(f"配体文件不存在: {ligand_pdbqt}")

    # 自动查找受体和配置
    if receptor_pdbqt is None:
        if target_name is None:
            raise ValueError("必须指定 receptor_pdbqt 或 target_name")
        receptor_pdbqt = get_receptor_pdbqt(target_name)
        if receptor_pdbqt is None:
            raise FileNotFoundError(f"找不到受体文件: {target_name}")

    if vina_config is None:
        if target_name is None:
            raise ValueError("必须指定 vina_config 或 target_name")
        vina_config = get_vina_config(target_name)
        if vina_config is None:
            raise FileNotFoundError(f"找不到配置文件: {target_name}")

    receptor_pdbqt = Path(receptor_pdbqt)
    vina_config = Path(vina_config)

    if not receptor_pdbqt.exists():
        raise FileNotFoundError(f"受体文件不存在: {receptor_pdbqt}")
    if not vina_config.exists():
        raise FileNotFoundError(f"配置文件不存在: {vina_config}")

    # 输出文件
    if output_file is None:
        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{ligand_pdbqt.stem}_docked.pdbqt"
    else:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Sim_4: Vina 分子对接")
        print(f"{'='*60}")
        print(f"配体: {ligand_pdbqt}")
        print(f"受体: {receptor_pdbqt}")
        print(f"配置: {vina_config}")
        print(f"输出: {output_file}")
        if exhaustiveness:
            print(f"详尽度: {exhaustiveness}（覆盖配置）")
        if num_modes:
            print(f"构象数: {num_modes}（覆盖配置）")

    # 构建 Vina 命令
    vina_path = config.TOOLS.get("vina", "vina")
    cmd = [
        vina_path,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--config", str(vina_config),
        "--out", str(output_file),
        "--cpu", "1"   # 强制单线程，防止容器OpenMP识别CPU=0卡死
    ]

    # 添加可选参数
    if exhaustiveness:
        cmd.extend(["--exhaustiveness", str(exhaustiveness)])
    if num_modes:
        cmd.extend(["--num_modes", str(num_modes)])

    if verbose:
        print(f"\n命令: {' '.join(cmd)}")
        print(f"超时: {timeout} 秒")
        print(f"\n开始对接...")

    # ========== Popen + OMP 环境设置 ==========
    import os
    proc = None
    stdout_data = ""
    stderr_data = ""
    # 复制当前环境，强制设置OMP_NUM_THREADS=1，双重规避OpenMP死锁
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            text=True,
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )
        
        # 实时读取输出（包括进度条）
        stdout_lines = []
        if verbose:
            print("\nVina Output:")
            print("-" * 60)
        
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip()
            stdout_lines.append(line)
            if verbose:
                print(line)  # 实时打印，包括进度条
        
        proc.stdout.close()
        returncode = proc.wait(timeout=timeout)
        
        stdout_data = '\n'.join(stdout_lines)
        stderr_data = ""  # 已合并到 stdout

        if verbose:
            print(f"\n返回码: {returncode}")
            if stdout_data:
                print(f"stdout:\n{stdout_data[:800]}")
            if stderr_data:
                print(f"stderr:\n{stderr_data[:800]}")

        if returncode != 0:
            error_msg = stderr_data[-500:] if stderr_data else "无错误信息"
            raise RuntimeError(f"Vina 对接失败, returncode={returncode}: {error_msg}")

        # 解析输出
        vina_results = parse_vina_output(stdout_data)
        if not vina_results:
            raise RuntimeError("无法解析 Vina stdout，未得到对接结果")

        best_energy = vina_results[0].binding_energy

        if verbose:
            print(f"\n{'='*60}")
            print(f"对接结果")
            print(f"{'='*60}")
            print(f"{'Rank':<6} {'Affinity (kcal/mol)':<20} {'RMSD l.b.':<12} {'RMSD u.b.':<12}")
            print("-" * 60)
            for r in vina_results:
                print(f"{r.rank:<6} {r.binding_energy:<20.4f} {r.rmsd_lb:<12.3f} {r.rmsd_ub:<12.3f}")
            print(f"{'='*60}")
            print(f"\n✓ 最佳结合能: {best_energy:.4f} kcal/mol")
            print(f"✓ 输出文件: {output_file}")
            print(f"{'='*60}\n")

        return VinaOutput(
            results=vina_results,
            output_file=output_file,
            best_energy=best_energy
        )

    except subprocess.TimeoutExpired:
        # 超时必须杀死进程，回收，避免僵尸进程
        if proc is not None:
            proc.kill()
            # 读取残留管道，释放缓冲区
            proc.communicate()
        raise RuntimeError(f"Vina 对接超时（{timeout} 秒）")

    except Exception as e:
        if proc is not None:
            try:
                proc.kill()
                proc.communicate()
            except Exception:
                pass
        raise RuntimeError(f"Vina 对接异常: {e}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vina 分子对接工具 (Sim_4)')
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='输入配体 PDBQT 文件')
    parser.add_argument('-t', '--target', type=str, default=None,
                        help='靶点名称（用于自动查找受体和配置）')
    parser.add_argument('-r', '--receptor', type=Path, default=None,
                        help='受体 PDBQT 文件路径')
    parser.add_argument('-c', '--config', type=Path, default=None,
                        help='Vina 配置文件路径')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='输出文件路径')
    parser.add_argument('-e', '--exhaustiveness', type=int, default=None,
                        help='搜索详尽度')
    parser.add_argument('-n', '--num-modes', type=int, default=None,
                        help='输出构象数量')
    parser.add_argument('--timeout', type=int, default=60,
                        help='超时时间（秒，默认 60）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式')
    
    args = parser.parse_args()
    
    try:
        output = run_vina_docking(
            ligand_pdbqt=args.input,
            target_name=args.target,
            receptor_pdbqt=args.receptor,
            vina_config=args.config,
            output_file=args.output,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            timeout=args.timeout,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(f"{output.best_energy:.4f}", end='')
        
        return 0
        
    except Exception as e:
        print(f"✗ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
