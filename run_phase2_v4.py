 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二主程序 V3 - 自适应MCTS-EGNN闭环优化（正确实现版）

核心逻辑：
1. MCTS 从根节点开始，逐层选择子节点
2. 当到达叶节点时，不断扩展直到序列完整（终端节点）
3. 终端节点被真正插入树中，EGNN 预测结合能
4. 奖励沿完整路径回传，更新所有节点的 Q 值
5. 定期从树中提取 Top-K 终端节点，用 Vina 验证
6. 用 Vina 结果微调 EGNN 模型
"""

import os
import sys
import json
import time
import random
import subprocess
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import config

# 导入MCTS模块
from peptide_state import PeptideState, create_root_node, MCTSNode
from selection import PUCTSelector
from expansion import ExpansionEngine
from simulation import SimulationEngine
from backpropagation import BackpropagationEngine
from seq_generator import generate_full_sequence, generate_n_random_fills

# 导入Vina对接
from vina import get_vina_paths, run_vina_with_progress
from ligand_generator import generate_ligand

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 导入EGNN
try:
    from egnn_predictor import create_egnn_predictor, EGNNPredictor

    HAS_EGNN = True
except ImportError:
    HAS_EGNN = False
    print("警告: EGNN模块不可用")

# 导入日志模块
try:
    from mcts_logger import init_logger, log_progress, log_debug, close_logger

    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False
    print("警告: 日志模块不可用")


def _gen_ligand_worker(args_tuple):
    """
    多进程worker顶层函数（必须全局，不能是类内局部函数，用于pickle序列化）
    args_tuple = (seq, crosslinker, crosslinker_positions, output_dir)
    返回 (seq, pdbqt_path | None)
    """
    seq, crosslinker, crosslinker_positions, output_dir = args_tuple
    try:
        pth = generate_ligand(
            sequence=seq,
            crosslinker=crosslinker,
            crosslinker_positions=crosslinker_positions,
            output_dir=output_dir
        )
        return seq, pth
    except Exception:
        return seq, None

class AdaptiveMCTSEngineV3:
    """
    自适应MCTS-EGNN闭环优化引擎 V3
    """

    def __init__(self, target_name: str):
        self.target_name = target_name
        # 目录设置
        self.results_dir = config.RESULTS_DIR / target_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # EGNN模型
        self.egnn_model = None
        self.egnn_model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"
        self.selector = PUCTSelector(c_puct=config.MCTS_CONFIG["c_puct"])
        self.backprop_engine = BackpropagationEngine(verbose=False)
        # 初始化 ExpansionEngine
        self.expansion_engine = ExpansionEngine(
            egnn_model=None,
            use_egnn_prior=False
        )
        # 全局候选池
        self.candidate_pool: Dict[str, float] = {}
        self.vina_validated: Dict[str, float] = {}
        self.test_set: Dict[str, float] = {}
        self.test_mae_history: List[float] = []
        # 统计信息
        self.egnn_round = 0
        self.total_mcts_iterations = 0
        self._pdbqt_cache = {}
        # ==========新增这一行 ==========
        self._egnn_energy_cache = {}  # EGNN预测能量缓存，避免重复推理
        # ==============================
        print(f"=" * 60)
        print(f"自适应MCTS引擎 V3 初始化")
        print(f"靶点: {target_name}")
        print(f"=" * 60)
        if HAS_LOGGER:
            init_logger(target_name)
            log_debug("engine", "自适应MCTS引擎 V3 初始化", {"target": target_name})

    # =================================================================
    # EGNN模型管理
    # =================================================================

    def load_egnn_model(self) -> bool:
        if not HAS_EGNN:
            print("错误: EGNN模块不可用")
            return False
        if not self.egnn_model_path.exists():
            print(f"错误: EGNN模型不存在: {self.egnn_model_path}")
            return False
        try:
            self.egnn_model = create_egnn_predictor()
            print(f"✓ EGNN模型加载成功")
            # 【新增】加载EGNN后，启用先验策略
            self.expansion_engine = ExpansionEngine(
                egnn_model=self.egnn_model,
                use_egnn_prior=True
            )
            print(f"✓ ExpansionEngine已切换为EGNN先验模式")
            return True
        except Exception as e:
            print(f"✗ EGNN模型加载失败: {e}")
            return False

    def collect_all_terminal_sequences(self, root: MCTSNode) -> List[Tuple[str, float]]:
        """
        遍历MCTS树，收集所有visit_count>0的终端节点，返回 (sequence, egnn_pred_energy)
        注意：不是MCTS的average_score，而是**EGNN预测energy**
        """
        candidates = []
        stack = [root]
        visited = set()
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            if node.is_terminal and node.visit_count > 0:
                seq = node.state.sequence
                try:
                    # 使用缓存读取EGNN预测energy，不再重新推理
                    e_energy = self.predict_with_egnn(seq)
                    candidates.append((seq, e_energy))
                except Exception as e:
                    print(f"[collect_terminal] skip seq {seq}, err:{e}")
            for child in node.children.values():
                if id(child) not in visited:
                    stack.append(child)
        return candidates

    def predict_with_egnn(self, sequence: str) -> float:
        # 先查能量缓存，命中直接返回，不跑推理
        if sequence in self._egnn_energy_cache:
            return self._egnn_energy_cache[sequence]

        if self.egnn_model is None:
            raise RuntimeError("EGNN模型未加载")
        from peptide_state import PeptideState
        state = PeptideState(sequence=sequence, crosslinker=config.CROSSLINKER)
        if state.sequence in self._pdbqt_cache:
            pdbqt_path = self._pdbqt_cache[state.sequence]
        else:
            pdbqt_path = generate_ligand(
                sequence=state.sequence,
                crosslinker=state.crosslinker or config.CROSSLINKER,
                crosslinker_positions=config.CROSSLINKER_POSITIONS
            )
            self._pdbqt_cache[state.sequence] = pdbqt_path
        energy = self.egnn_model.predict(pdbqt_path)
        # 存入能量缓存
        self._egnn_energy_cache[sequence] = energy
        return energy

    def batch_predict_with_egnn(self, sequences: List[str]) -> List[float]:
        energies = []
        for seq in sequences:
            try:
                energy = self.predict_with_egnn(seq)
                energies.append(energy)
            except Exception as e:
                print(f"  预测失败 {seq}: {e}")
                energies.append(0.0)
        return energies

    # =================================================================
    # 冷启动
    # =================================================================

    def cold_start(self, n_sequences: int = 1500) -> bool:
        print("\n" + "=" * 60)
        print("冷启动：生成初始数据【提速版：批量预生成配体 + 并行对接，分批落盘】")
        print("=" * 60)
        print(f"\n[1/4] 生成{n_sequences}个随机序列...")
        sequences = generate_n_random_fills(config.PEPTIDE_TEMPLATE, n_sequences)
        print(f"  ✓ 生成完成")

        print(f"\n[2/4] 多进程批量预生成全部配体PDBQT...")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from vina import get_vina_paths

        vina_paths = get_vina_paths(self.target_name)
        output_dir = config.BASE_DIR / "temp" / "ligand_generator"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 组装参数元组，传给顶层worker函数
        task_args = [
            (seq, config.CROSSLINKER, config.CROSSLINKER_POSITIONS, output_dir)
            for seq in sequences
        ]

        n_workers = config.PARALLEL_VINA_CONFIG["num_workers"]
        gen_results = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for seq, pth in executor.map(_gen_ligand_worker, task_args):
                if pth is not None:
                    gen_results[seq] = pth
                    self._pdbqt_cache[seq] = pth

        valid_seqs = list(gen_results.keys())
        print(f"  配体生成成功: {len(valid_seqs)}/{n_sequences}")
        if len(valid_seqs) < 51:
            print("✗ 成功生成配体的序列太少，冷启动失败")
            return False

        # -------- 加载历史energies.csv，恢复到内存字典，防止覆盖旧数据 --------
        energies_file = self.results_dir / "energies.csv"
        all_vina_results = dict()
        if energies_file.exists():
            import csv
            with open(energies_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        s = row["sequence"]
                        e = float(row["energy"])
                        all_vina_results[s] = e
                    except Exception:
                        pass
            print(f"  ✓ 加载历史energies.csv，已有 {len(all_vina_results)} 条记录")

        # 过滤：已经在历史csv中的序列不再重复对接
        todo_seqs = [s for s in valid_seqs if s not in all_vina_results]
        print(f"  需要执行对接的新序列数量: {len(todo_seqs)}")

        print(f"\n[3/4] 并行GNINA对接（冷启动低精度模式，关闭对接校验，每{str(n_workers)}个完成即写csv）...")
        orig_exhaust = config.VINA_CONFIG["exhaustiveness"]
        orig_num_modes = config.VINA_CONFIG["num_modes"]
        try:
            config.VINA_CONFIG["exhaustiveness"] = 3
            config.VINA_CONFIG["num_modes"] = 2

            from vina import vina_dock

            future_map = dict()
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # 全部提交任务
                for seq in todo_seqs:
                    fut = executor.submit(
                        vina_dock,
                        sequence=seq,
                        target_name=self.target_name,
                        validate_docking=False,
                        verbose=False
                    )
                    future_map[fut] = seq

                done_cnt = 0
                # as_completed：每完成一个future就返回
                for fut in as_completed(future_map):
                    seq = future_map[fut]
                    done_cnt += 1
                    try:
                        energy = fut.result()
                        # 仅合格才存入内存总表
                        if energy != 0.0 and energy < -3.0:
                            all_vina_results[seq] = energy
                    except Exception as e:
                        print(f"[对接失败] {seq}, err:{e}")

                    # ====== 关键：每完成 n_workers 个任务，写入一次csv ======
                    if done_cnt % n_workers == 0:
                        import csv
                        print(f"[分批保存] 已完成 {done_cnt}/{len(todo_seqs)}，写入energies.csv")
                        with open(energies_file, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["sequence", "energy"])
                            for s, e_val in all_vina_results.items():
                                writer.writerow([s, e_val])

                # 循环结束，把剩余不足一批的结果强制落盘一次
                import csv
                print(f"[最终保存]全部对接任务结束，总有效记录 {len(all_vina_results)}")
                with open(energies_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["sequence", "energy"])
                    for s, e_val in all_vina_results.items():
                        writer.writerow([s, e_val])

        finally:
            config.VINA_CONFIG["exhaustiveness"] = orig_exhaust
            config.VINA_CONFIG["num_modes"] = orig_num_modes

        # 转为list给后续训练使用
        vina_results = [(s, e) for s, e in all_vina_results.items()]
        print(f"  ✓ Vina完成: {len(vina_results)} 有效记录（energy < -3.0）")

        if len(vina_results) < 51:
            print("✗ 有效对接结果太少，冷启动失败")
            return False

        print(f"\n[4/4] 划分数据集 (8:1:1)...")
        train_data, val_data, test_data = self._split_dataset(vina_results, 0.8, 0.1, 0.1)
        print(f"  ✓ 训练: {len(train_data)} / 验证: {len(val_data)} / 测试: {len(test_data)}")
        self.test_set = dict(test_data)
        self.vina_validated = dict(vina_results)

        print(f"\n[5/5] 训练初始EGNN模型...")
        success = self._train_egnn(train_data + val_data)
        if not success:
            return False
        self._evaluate_test_set()
        self.egnn_round = 1

        print("\n" + "=" * 60)
        print(f"冷启动完成！EGNN第{self.egnn_round}轮")
        print("=" * 60)
        return True

    def _run_vina_batch(self, sequences: List[str]) -> List[Tuple[str, float]]:
        """
        !!! 本函数现在不再被cold_start调用，保留仅作兼容；
        cold_start 上面直接使用 batch_vina_dock 并行路径。
        MCTS闭环内部不使用该函数。
        """
        vina_paths = get_vina_paths(self.target_name)
        existing_results = {}
        energies_file = self.results_dir / "energies.csv"
        if energies_file.exists():
            import csv
            with open(energies_file, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            existing_results[row[0]] = float(row[1])
                        except ValueError:
                            continue

        results = []
        for seq in sequences:
            if seq in existing_results and existing_results[seq] < -3.0:
                results.append((seq, existing_results[seq]))
                continue
            try:
                if seq in self._pdbqt_cache:
                    pdbqt_path = self._pdbqt_cache[seq]
                else:
                    pdbqt_path = generate_ligand(
                        sequence=seq,
                        crosslinker=config.CROSSLINKER,
                        crosslinker_positions=config.CROSSLINKER_POSITIONS
                    )
                    self._pdbqt_cache[seq] = pdbqt_path

                from vina import vina_dock
                e = vina_dock(
                    sequence=seq,
                    target_name=self.target_name,
                    validate_docking=True,
                    verbose=False
                )
                if e != 0.0 and e < -3.0:
                    results.append((seq, e))
            except Exception:
                continue
        return results

        def flush_results(results: List[Tuple[str, float]], is_final: bool = False):
            import csv
            all_results = existing_results.copy()
            for seq, energy in results:
                all_results[seq] = energy

            with open(energies_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sequence', 'energy'])
                for seq, energy in all_results.items():
                    writer.writerow([seq, energy])

            if not is_final:
                print(f"    【保存】已写入 {len(all_results)} 个结果")

        total_sequences = len(sequences)
        batch_size = 10

        for i, seq in enumerate(sequences, 1):
            if seq in existing_results and existing_results[seq] < -3.0:
                print(f"  进度: {i}/{total_sequences} 【跳过】{seq}: 已存在 (energy={existing_results[seq]:.2f})")
                results.append((seq, existing_results[seq]))
                continue

            if i % 1 == 0:
                print(f"  进度: {i}/{total_sequences}")

            try:
                if seq in self._pdbqt_cache:
                    pdbqt_path = self._pdbqt_cache[seq]
                else:
                    pdbqt_path = generate_ligand(
                        sequence=seq,
                        crosslinker=config.CROSSLINKER,
                        crosslinker_positions=config.CROSSLINKER_POSITIONS
                    )
                    self._pdbqt_cache[seq] = pdbqt_path

                result = run_vina_with_progress(
                    ligand_pdbqt=pdbqt_path,
                    receptor_pdbqt=vina_paths['receptor'],
                    vina_config=vina_paths['config'],
                    n_cpu=config.VINA_CONFIG.get("cpu", 4),
                    #pocket_center=pocket_center,
                    validate_docking=True,
                    verbose=False,
                    sequence=seq,
                    target_name=self.target_name
                )

                if result.success and result.binding_energy < -3.0:
                    print(f"    【成功】{seq}: 结合能={result.binding_energy:.2f} kcal/mol")
                    results.append((seq, result.binding_energy))
                    if HAS_LOGGER:
                        log_debug("vina", f"Vina对接成功", {
                            "sequence": seq,
                            "binding_energy": result.binding_energy,
                            "progress": f"{i}/{total_sequences}"
                        })
                elif result.success:
                    print(f"    【过滤】{seq}: 结合能={result.binding_energy:.2f} (>= -3.0，太弱)")
                else:
                    error_msg = result.error_message if result.error_message else "未知错误"
                    print(f"    【失败】{seq}: {error_msg}")

            except Exception as e:
                print(f"    【异常】{seq}: {e}")
                import traceback
                traceback.print_exc()

            if i % batch_size == 0:
                flush_results(results, is_final=False)

        flush_results(results, is_final=True)

        if HAS_LOGGER:
            log_debug("vina", f"Vina批次完成", {
                "total": len(sequences),
                "success": len(results),
                "success_rate": len(results) / len(sequences) if sequences else 0
            })

        return results

    def _split_dataset(self, data: List[Tuple[str, float]],
                       train_ratio: float, val_ratio: float, test_ratio: float):
        import random
        random.shuffle(data)

        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]

        return train_data, val_data, test_data

    def _train_egnn(self, data: List[Tuple[str, float]], n_epochs: int = 100) -> bool:
        if not HAS_EGNN:
            return False

        try:
            sequences_file = config.BASE_DIR / "sequences.txt"
            energies_file = self.results_dir / "energies.csv"

            with open(sequences_file, 'w') as f:
                for seq, _ in data:
                    f.write(f"{seq}\n")

            with open(energies_file, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['sequence', 'energy'])
                for seq, energy in data:
                    writer.writerow([seq, energy])

            print("  [1/2] 准备EGNN数据 (EGNN_1.py)...")
            result_prep = subprocess.run(
                [
                    sys.executable, "EGNN_1.py",
                    "-s", str(sequences_file),
                    "-e", str(energies_file),
                    "--target", self.target_name
                ],
                cwd=config.BASE_DIR,
                capture_output=True,
                text=True,
                timeout=3600
            )

            if result_prep.returncode != 0:
                print(f"  ✗ 数据准备失败: {result_prep.stderr}")
                if HAS_LOGGER:
                    log_debug("egnn_prep", f"EGNN数据准备失败", {
                        "error": result_prep.stderr,
                        "n_data": len(data)
                    })
                return False

            print("  ✓ 数据准备完成")

            print("  [2/2] 训练EGNN模型 (EGNN_23.py)...")
            result = subprocess.run(
                [
                    sys.executable, "egnn_23.py",
                    "--target", self.target_name
                ],
                cwd=config.BASE_DIR,
                capture_output=True,
                text=True,
                timeout=3600
            )
            if result.returncode == 0:
                print("  ✓ EGNN训练完成")
                if HAS_LOGGER:
                    log_debug("egnn_train", f"EGNN训练完成", {
                        "n_epochs": n_epochs,
                        "n_data": len(data),
                        "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                    })
                return self.load_egnn_model()
            else:
                print(f"  ✗ 训练失败: {result.stderr}")
                if HAS_LOGGER:
                    log_debug("egnn_train", f"EGNN训练失败", {
                        "error": result.stderr,
                        "n_data": len(data)
                    })
                return False

        except Exception as e:
            print(f"  ✗ EGNN训练失败: {e}")
            return False

    def _evaluate_test_set(self):
        if not self.egnn_model or not self.test_set:
            return

        sequences = list(self.test_set.keys())
        true_energies = list(self.test_set.values())
        pred_energies = self.batch_predict_with_egnn(sequences)

        mae = np.mean([abs(p - t) for p, t in zip(pred_energies, true_energies)])
        rmse = np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(pred_energies, true_energies)]))

        ss_res = np.sum([(p - t) ** 2 for p, t in zip(pred_energies, true_energies)])
        ss_tot = np.sum([(t - np.mean(true_energies)) ** 2 for t in true_energies])
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if len(pred_energies) > 1:
            pearson_r = np.corrcoef(pred_energies, true_energies)[0, 1]
        else:
            pearson_r = 0.0

        self.test_mae_history.append(mae)

        print(f"  测试集评估:")
        print(f"    MAE:  {mae:.3f} kcal/mol")
        print(f"    RMSE: {rmse:.3f} kcal/mol")
        print(f"    R²:   {r2:.3f}")
        print(f"    Pearson r: {pearson_r:.3f}")

        if HAS_LOGGER:
            log_debug("egnn_eval", f"EGNN测试集评估", {
                "egnn_round": self.egnn_round,
                "test_set_size": len(sequences),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "pearson_r": float(pearson_r)
            })

    # =================================================================
    # 核心：MCTS 迭代（修复版）
    # =================================================================

    def _force_complete(self, partial_sequence: str) -> str:
        """把 'x' 用随机氨基酸填满，保留已填的"""
        seq_list = list(partial_sequence)
        for i, c in enumerate(seq_list):
            if c in ['x', 'X', '_']:
                seq_list[i] = random.choice(config.ALLOWED_AMINO_ACIDS)
        return ''.join(seq_list)

    def _create_terminal_node(self, full_seq: str, parent: MCTSNode) -> MCTSNode:
        """创建终端节点并插入树中"""
        terminal_state = PeptideState(
            sequence=full_seq,
            crosslinker=config.CROSSLINKER,
            disulfide_bonds=[]
        )
        terminal_state.is_sequence_complete = True
        terminal_state.is_topology_complete = True

        terminal_node = MCTSNode(
            state=terminal_state,
            parent=parent,
            prior_prob=1.0,
            decision_level=1,
            decision_action=full_seq[-1] if full_seq else "sim"
        )
        parent.children[full_seq[-1] + "_term"] = terminal_node
        return terminal_node

    def mcts_iteration(self, root: MCTSNode) -> MCTSNode:
        """
        执行一次完整的 MCTS 迭代，确保到达终端节点

        关键修复：
        1. Selection → Expansion 循环，直到终端节点被创建并插入树中
        2. 终端节点的奖励沿完整路径回传
        """
        # 一、Selection: 从根走到当前最深的叶节点
        path = self.selector.select_path(
            root,
            can_expand_fn=lambda node: self.expansion_engine.can_expand(node)
        )
        leaf = path[-1]

        # 二、Expansion + 强制补全：从 leaf 开始，不断扩展直到终端
        while not leaf.is_terminal:
            # 检查是否可以继续扩展（还有可选的氨基酸）
            if self.expansion_engine.can_expand(leaf):

                # 【修复】max_expansions调大，一次生成一批候选子节点
                new_children = self.expansion_engine.expand(leaf, max_expansions=10)
                # ===== 这里插入DEBUG打印 =====
                print(f"[DEBUG expand] partial={leaf.state.sequence}, new actions={list(new_children.keys())}")
                if new_children:
                    child_list = list(new_children.values())
                    if self.expansion_engine.use_egnn_prior:
                        # 有EGNN先验：使用PUCT从这批新生成子节点选一个
                        child = self.selector.select(leaf)
                    else:
                        # 无EGNN先验(冷启动前期)：随机采样，避免永远选列表第一个A
                        import random
                        child = random.choice(child_list)
                    path.append(child)
                    leaf = child
                    continue
                else:
                    # 无可用氨基酸，强制补全到终端
                    full_seq = self._force_complete(leaf.state.sequence)
                    terminal_node = self._create_terminal_node(full_seq, leaf)
                    path.append(terminal_node)
                    leaf = terminal_node
                    break
            else:
                # 无法扩展，强制补全
                full_seq = self._force_complete(leaf.state.sequence)
                terminal_node = self._create_terminal_node(full_seq, leaf)
                path.append(terminal_node)
                leaf = terminal_node
                break

        # 三、Simulation: 现在 leaf 一定是终端节点
        energy = self.predict_with_egnn(leaf.state.sequence)
        reward = self._energy_to_reward(energy)

        # 四、Backpropagation: 把奖励沿完整路径回传
        self.backprop_engine.backpropagate(path, reward)

        return root

    def _energy_to_reward(self, energy: float) -> float:
        """将结合能转换为 [0, 1] 奖励值"""
        reward = 1.0 - (energy / -15.0)
        return max(0.0, min(1.0, reward))

    def _heuristic_score(self, node: MCTSNode) -> float:
        """启发式分数（备用）"""
        seq = node.state.sequence
        completed = sum(1 for c in seq if c not in ['_', 'x', 'X'])
        total = len(seq)
        return 0.3 + 0.5 * (completed / total)

    def extract_top_candidates(self, root: MCTSNode, top_n: int = 100) -> List[Tuple[str, float]]:
        """
        从 MCTS 树中提取 Top-N 终端节点（按 Q 值排序）
        """
        candidates = []
        stack = [root]
        visited = set()

        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)

            if node.is_terminal and node.visit_count > 0:
                candidates.append((node.state.sequence, node.average_score))

            for child in node.children.values():
                if id(child) not in visited:
                    stack.append(child)

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def run(self,
            total_iterations: int = 50000,
            validation_interval: int = 1000,
            validation_top_n: int = 100) -> None:
        """
        MCTS‑EGNN闭环：每validation_interval次MCTS迭代
        1. 收集全部终端序列，使用EGNN预测energy排序
        2. 选出EGNN打分最优top‑N，再交给GNINA/Vina做真实对接
        """
        print(f"\n开始 MCTS‑EGNN 闭环优化")
        print(f"  总迭代次数: {total_iterations}")
        print(f"  每 {validation_interval} MCTS迭代 → EGNN筛选top {validation_top_n} 做Vina/GNINA验证")
        print("=" * 60)
        root = create_root_node()
        validated_sequences = set()

        for iteration in range(validation_interval, total_iterations + 1, validation_interval):
            # 执行validation_interval次MCTS迭代
            for _ in range(validation_interval):
                root = self.mcts_iteration(root)
                self.total_mcts_iterations += 1

            print(f"\n{'=' * 60}")
            print(f"迭代 {iteration}/{total_iterations}: 收集全部终端序列，EGNN打分筛选")
            print(f"{'=' * 60}")

            # ==========【修改核心】收集全部终端序列，用EGNN预测值筛选top ==========
            all_term_seqs = self.collect_all_terminal_sequences(root)
            print(f"  MCTS树内有效终端序列总数: {len(all_term_seqs)}")
            if len(all_term_seqs) == 0:
                print("  警告：没有终端序列，跳过本轮验证")
                continue

            # ✅按EGNN预测energy升序：数值越小结合能力越好，取前validation_top_n
            all_term_seqs.sort(key=lambda x: x[1])
            # 取EGNN预测最优前N个
            egnn_top_candidates = all_term_seqs[:validation_top_n]
            print(f"  EGNN预测筛选出top‑{validation_top_n}序列")

            print("====EGNN筛选Top序列（seq | egnn_pred_energy）====")
            for seq, e in egnn_top_candidates[:10]:
                print(f"{seq:12s} | {e:.3f}")

            # 过滤掉已经Vina验证过的序列
            new_candidates = [
                (seq, e_energy)
                for seq, e_energy in egnn_top_candidates
                if seq not in validated_sequences
            ]
            if not new_candidates:
                print("  EGNN top候选全部已经验证过，跳过Vina对接")
                continue
            print(f"  其中 {len(new_candidates)} 个是未验证序列，执行GNINA对接")

            # ========== 对EGNN筛选出来的候选执行GNINA真实对接 ==========
            vina_results = []
            for seq, egnn_pred_e in new_candidates:
                real_energy = self._vina_dock(seq)
                if real_energy is not None and real_energy < 0:
                    vina_results.append((seq, real_energy))
                    validated_sequences.add(seq)

            print(f"  GNINA/Vina验证成功: {len(vina_results)} 个")
            if vina_results:
                self._update_training_data(vina_results)
                self._finetune_egnn()
                self._evaluate_test_set()

        print(f"\n{'=' * 60}")
        print("MCTS‑EGNN 闭环优化完成！")
        print(f"  总迭代次数: {total_iterations}")
        print(f"  已验证序列数: {len(validated_sequences)}")
        print(f"  候选池大小: {len(self.candidate_pool)}")
        print(f"{'=' * 60}")

    def _vina_dock(self, sequence: str) -> Optional[float]:
        """
        直接调用gnina二进制，完全绕开vina.py，规避口袋验证KDTree死锁；输出写入临时日志文件，无PIPE管道阻塞
        """
        import subprocess
        import re
        import tempfile
        import os
        try:
            vina_paths = get_vina_paths(self.target_name)
            receptor_pdb = vina_paths["receptor"]
            vina_cfg_path = vina_paths["config"]

            # 复用pdbqt缓存
            if sequence in self._pdbqt_cache:
                pdbqt_path = self._pdbqt_cache[sequence]
            else:
                pdbqt_path = generate_ligand(
                    sequence=sequence,
                    crosslinker=config.CROSSLINKER,
                    crosslinker_positions=config.CROSSLINKER_POSITIONS
                )
                self._pdbqt_cache[sequence] = pdbqt_path

            # 解析盒子参数
            # 解析盒子参数
            cx = cy = cz = None
            sx = sy = sz = None
            with open(vina_cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val_str = line.split("=", 1)
                    key = key.strip()
                    val_str = val_str.strip()
                    # 仅解析这6个数值字段，其余全部跳过
                    if key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"):
                        val = float(val_str)
                        if key == "center_x":
                            cx = val
                        elif key == "center_y":
                            cy = val
                        elif key == "center_z":
                            cz = val
                        elif key == "size_x":
                            sx = val
                        elif key == "size_y":
                            sy = val
                        elif key == "size_z":
                            sz = val
            # 校验必须全部读到
            if cx is None or cy is None or cz is None or sx is None or sy is None or sz is None:
                raise RuntimeError(
                    f"vina_config.txt 缺失盒子参数!\n"
                    f"cx={cx}, cy={cy}, cz={cz} | sx={sx}, sy={sy}, sz={sz}"
                )


            # 组装gnina命令，与vina.py保持一致参数
            cmd = [
                "gnina",
                "-r", str(receptor_pdb),
                "-l", str(pdbqt_path),
                "--center_x", str(cx),
                "--center_y", str(cy),
                "--center_z", str(cz),
                "--size_x", str(sx),
                "--size_y", str(sy),
                "--size_z", str(sz),
                "--exhaustiveness", "8",
                "--num_modes", "5",
                "--cnn_scoring", "rescore",
                "--no_gpu"
            ]
            # 临时日志，全部输出写入文件，不使用PIPE
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as log_f:
                log_path = log_f.name
            try:
                with open(log_path, "w") as f_out:
                    proc = subprocess.run(cmd, stdout=f_out, stderr=f_out, timeout=600)
                # 读取输出解析mode1 affinity
                with open(log_path, "r", encoding="utf-8") as f_log:
                    log_text = f_log.read()
                best_energy = None
                in_table = False
                for line in log_text.splitlines():
                    line = line.strip()
                    if "mode |" in line:
                        in_table = True
                        continue
                    if "-----+" in line:
                        continue
                    if in_table and line and line[0].isdigit():
                        parts = line.split()
                        if len(parts) >= 2:
                            best_energy = float(parts[1])
                            break
                if proc.returncode == 0 and best_energy is not None:
                    print(f"  [GNINA‑raw] {sequence} best affinity = {best_energy:.2f}")
                    return best_energy
                else:
                    print(f"  [GNINA‑raw] seq {sequence} failed, retcode={proc.returncode}")
                    return None
            finally:
                if os.path.exists(log_path):
                    os.unlink(log_path)
        except Exception as e:
            print(f"  [GNINA‑raw]对接异常 {sequence[:20]}...: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _update_training_data(self, vina_results: List[Tuple[str, float]]):
        import csv
        from datetime import datetime

        dataset_path = self.results_dir / "dataset.csv"
        timestamp = datetime.now().isoformat()

        for seq, energy in vina_results:
            self.vina_validated[seq] = energy
            self.candidate_pool[seq] = energy

        with open(dataset_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for seq, energy in vina_results:
                writer.writerow([seq, config.CROSSLINKER, '', energy, 'vina', timestamp])

    def _finetune_egnn(self, n_epochs: int = 20):
        if not self.vina_validated:
            print("  没有新数据，跳过微调")
            return

        print(f"  微调 EGNN（使用 {len(self.vina_validated)} 个数据，{n_epochs} epochs）...")
        data = list(self.vina_validated.items())
        self._train_egnn(data, n_epochs=n_epochs)


# =================================================================
# 主程序入口
# =================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="自适应MCTS-EGNN闭环优化 V3")
    parser.add_argument("--max-iterations", type=int, default=50000, help="MCTS总迭代次数")
    parser.add_argument("--validation-interval", type=int, default=1000, help="验证间隔")
    parser.add_argument("--validation-top-n", type=int, default=100, help="每次验证提取的候选数")
    parser.add_argument("-t", "--target", required=True, help="靶点名称（如1LYZ）")
    parser.add_argument("--cold-start-n", type=int, default=1500, help="冷启动序列数")

    args = parser.parse_args()

    engine = AdaptiveMCTSEngineV3(args.target)

    if engine.egnn_model_path.exists():
        print(f"\n检测到已有EGNN模型，跳过冷启动")
        engine.load_egnn_model()
    else:
        print(f"\n未检测到EGNN模型，执行冷启动...")
        if not engine.cold_start(n_sequences=args.cold_start_n):
            print("冷启动失败！")
            sys.exit(1)

    engine.run(
        total_iterations=args.max_iterations,
        validation_interval=args.validation_interval,
        validation_top_n=args.validation_top_n
    )

    if HAS_LOGGER:
        close_logger()


if __name__ == "__main__":
    main()