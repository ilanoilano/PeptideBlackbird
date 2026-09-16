#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二主程序 V3 - 自适应MCTS-EGNN闭环优化（正确实现版）

根据对话整理的正确流程：
1. 每轮选择f(N)个母节点（N=EGNN迭代轮次）
2. 每个母节点生成g(N)个随机填充，EGNN评估得平均亲和度
3. Softmax分配f(N+1)个名额（按平均亲和度绝对值）
4. 每个母节点扩展Top-k子节点（k=分配名额，EGNN预测选Top-k，绝不随机）
5. 收集终端节点，选Top-h(N)个Vina验证
6. 8:1:1划分，微调EGNN
7. 重复直到收敛

关键函数：
- f(N) = min(19 + (N-1)*2, 100)  # 母节点数，递增
- g(N) = max(50 - (N-1)*2, 10)   # 随机填充数，递减
- h(N) = min(40 + (N-1)*3, 200)  # Vina验证数，递增
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
from vina import get_vina_paths, run_vina_with_progress, get_pocket_center
from ligand_generator import generate_ligand

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


class AdaptiveMCTSEngineV3:
    """
    自适应MCTS-EGNN闭环优化引擎 V3
    
    正确实现：
    - 全局候选池保存所有历史终端节点
    - Softmax分配名额
    - EGNN批量预测选Top-k（绝不随机）
    - f(N), g(N), h(N)动态调整
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

        # 【新增】初始化 ExpansionEngine（即使没有 EGNN 模型也要初始化）
        # 原因：MCTS 的 Expansion 阶段需要它来：
        #   1. 获取下一个可变位置
        #   2. 获取该位置允许的氨基酸列表
        #   3. 填充氨基酸生成子节点
        # 这些功能不依赖 EGNN 模型，只需要模板配置
        self.expansion_engine = ExpansionEngine(
            egnn_model=None,  # 没有 EGNN 模型也能工作（均匀分布）
            use_egnn_prior=False  # 不使用 EGNN 先验
        )
        
        # 全局候选池（关键：保存所有历史终端节点）
        self.candidate_pool: Dict[str, float] = {}  # {sequence: egnn_score}
        
        # 已Vina验证的序列
        self.vina_validated: Dict[str, float] = {}  # {sequence: vina_energy}
        
        # 测试集管理
        self.test_set: Dict[str, float] = {}
        self.test_mae_history: List[float] = []
        
        # 统计信息
        self.egnn_round = 0  # EGNN迭代轮次
        self.total_mcts_iterations = 0
        
        # 【新增】PDBQT 缓存，避免重复生成分子
        self._pdbqt_cache = {}  # {sequence: pdbqt_path}
        
        # 引擎组件
        self.selector = PUCTSelector(c_puct=config.MCTS_CONFIG["c_puct"])
        self.backprop_engine = BackpropagationEngine(verbose=False)
        
        print(f"="*60)
        print(f"自适应MCTS引擎 V3 初始化")
        print(f"靶点: {target_name}")
        print(f"="*60)
        
        # 初始化日志
        if HAS_LOGGER:
            init_logger(target_name)
            log_debug("engine", "自适应MCTS引擎 V3 初始化", {"target": target_name})
    
    # =================================================================
    # 动态参数函数 f(N), g(N), h(N)
    # 从 config 读取配置，确保修改 config 能生效
    # =================================================================
    

    
    # =================================================================
    # EGNN模型管理
    # =================================================================
    
    def load_egnn_model(self) -> bool:
        """加载EGNN模型"""
        if not HAS_EGNN:
            print("错误: EGNN模块不可用")
            return False
        
        if not self.egnn_model_path.exists():
            print(f"错误: EGNN模型不存在: {self.egnn_model_path}")
            return False
        
        try:
            self.egnn_model = create_egnn_predictor()
            print(f"✓ EGNN模型加载成功")
            return True
        except Exception as e:
            print(f"✗ EGNN模型加载失败: {e}")
            return False
    
    def predict_with_egnn(self, sequence: str) -> float:
        """使用EGNN预测单个序列的亲和度"""
        if self.egnn_model is None:
            raise RuntimeError("EGNN模型未加载")
        
        from peptide_state import PeptideState
        state = PeptideState(sequence=sequence, crosslinker=config.CROSSLINKER)
        
        # 【新增】检查缓存
        if state.sequence in self._pdbqt_cache:
            pdbqt_path = self._pdbqt_cache[state.sequence]
            print(f"  【缓存】使用已生成的配体: {pdbqt_path.name}")
        else:
            # 生成PDBQT并预测
            pdbqt_path = generate_ligand(
                sequence=state.sequence,
                crosslinker=state.crosslinker or config.CROSSLINKER,
                crosslinker_positions=config.CROSSLINKER_POSITIONS
            )
            # 缓存生成的 PDBQT 路径
            self._pdbqt_cache[state.sequence] = pdbqt_path
        
        return self.egnn_model.predict(pdbqt_path)
    
    def batch_predict_with_egnn(self, sequences: List[str]) -> List[float]:
        """批量预测多个序列的亲和度"""
        energies = []
        for seq in sequences:
            try:
                energy = self.predict_with_egnn(seq)
                energies.append(energy)
            except Exception as e:
                print(f"  预测失败 {seq}: {e}")
                energies.append(0.0)  # 失败时返回0
        return energies
    
    # =================================================================
    # 冷启动
    # =================================================================
    
    def cold_start(self, n_sequences: int = 1500) -> bool:
        """
        冷启动：生成初始数据并训练EGNN
        
        Args:
            n_sequences: 初始序列数量（默认1500）
        """
        print("\n" + "="*60)
        print("冷启动：生成初始数据")
        print("="*60)
        
        # 步骤1: 生成随机序列
        print(f"\n[1/4] 生成{n_sequences}个随机序列...")
        # 使用 config.PEPTIDE_TEMPLATE 作为模板
        sequences = generate_n_random_fills(config.PEPTIDE_TEMPLATE, n_sequences)
        print(f"  ✓ 生成完成")
        
        # 步骤2: Vina对接
        print(f"\n[2/4] Vina对接（这可能需要较长时间）...")
        vina_results = self._run_vina_batch(sequences)
        print(f"  ✓ Vina完成: {len(vina_results)}/{len(sequences)} 成功")
        
        if len(vina_results) < 51:
            print("✗ 成功对接的序列太少，冷启动失败")
            return False
        
        # 步骤3: 8:1:1划分
        print(f"\n[3/4] 划分数据集 (8:1:1)...")
        train_data, val_data, test_data = self._split_dataset(vina_results, 0.8, 0.1, 0.1)
        print(f"  ✓ 训练: {len(train_data)} / 验证: {len(val_data)} / 测试: {len(test_data)}")
        
        # 保存测试集
        self.test_set = dict(test_data)
        self.vina_validated = dict(vina_results)
        
        # 步骤4: 训练EGNN
        print(f"\n[4/4] 训练初始EGNN模型...")
        success = self._train_egnn(train_data + val_data)
        if not success:
            return False
        
        # 评估测试集
        self._evaluate_test_set()
        
        self.egnn_round = 1
        print("\n" + "="*60)
        print(f"冷启动完成！EGNN第{self.egnn_round}轮")
        print("="*60)
        return True

    def _run_vina_batch(self, sequences: List[str]) -> List[Tuple[str, float]]:
        """批量运行Vina对接，每10个自动保存一次"""
        vina_paths = get_vina_paths(self.target_name)

        # 【修复】获取口袋中心坐标，用于验证
        pocket_center = get_pocket_center(self.target_name)
        if pocket_center is not None:
            print(f"  口袋中心: ({pocket_center[0]:.3f}, {pocket_center[1]:.3f}, {pocket_center[2]:.3f})")
        else:
            print(f"  【警告】无法获取口袋中心，验证将跳过位置检查")

        results = []
        energies_file = self.results_dir / "energies.csv"

        # 【新增】如果文件已存在，读取已有结果避免重复对接
        existing_results = {}
        if energies_file.exists():
            import csv
            with open(energies_file, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # 跳过表头
                for row in reader:
                    if len(row) >= 2:
                        try:
                            existing_results[row[0]] = float(row[1])
                        except ValueError:
                            continue
            print(f"  【恢复】已存在 {len(existing_results)} 个对接结果，将跳过已对接的序列")

        # 【新增】保存函数：每10个写入一次
        def flush_results(results: List[Tuple[str, float]], is_final: bool = False):
            """将结果写入 energies.csv"""
            import csv
            # 合并已有结果和新结果
            all_results = existing_results.copy()
            for seq, energy in results:
                all_results[seq] = energy

            # 写入文件
            with open(energies_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sequence', 'energy'])
                for seq, energy in all_results.items():
                    writer.writerow([seq, energy])

            if not is_final:
                print(f"    【保存】已写入 {len(all_results)} 个结果到 {energies_file}")

        total_sequences = len(sequences)
        batch_size = 10
        valid_count = 0

        for i, seq in enumerate(sequences, 1):
            # 【新增】跳过已对接的序列
            if seq in existing_results and existing_results[seq] < -3.0:
                print(f"  进度: {i}/{total_sequences} 【跳过】{seq}: 已存在 (energy={existing_results[seq]:.2f})")
                results.append((seq, existing_results[seq]))
                valid_count += 1
                continue

            if i % 1 == 0:
                print(f"  进度: {i}/{total_sequences}")

            try:
                # 【新增】检查缓存
                if seq in self._pdbqt_cache:
                    pdbqt_path = self._pdbqt_cache[seq]
                    print(f"    【缓存】使用已生成的配体: {pdbqt_path.name}")
                else:
                    pdbqt_path = generate_ligand(
                        sequence=seq,
                        crosslinker=config.CROSSLINKER,
                        crosslinker_positions=config.CROSSLINKER_POSITIONS
                    )
                    # 缓存生成的 PDBQT 路径
                    self._pdbqt_cache[seq] = pdbqt_path

                # 【修复】传递 pocket_center 和 validate_docking 参数
                result = run_vina_with_progress(
                    ligand_pdbqt=pdbqt_path,
                    receptor_pdbqt=vina_paths['receptor'],
                    vina_config=vina_paths['config'],
                    n_cpu=config.VINA_CONFIG.get("cpu", 4),
                    pocket_center=pocket_center,
                    validate_docking=True,
                    verbose=False,
                    sequence=seq,
                    target_name=self.target_name
                )

                if result.success and result.binding_energy < -3.0:
                    print(f"    【成功】{seq}: 结合能={result.binding_energy:.2f} kcal/mol")
                    results.append((seq, result.binding_energy))
                    valid_count += 1
                    if HAS_LOGGER:
                        log_debug("vina", f"Vina对接成功", {
                            "sequence": seq,
                            "binding_energy": result.binding_energy,
                            "progress": f"{i}/{total_sequences}"
                        })
                elif result.success:
                    print(f"    【过滤】{seq}: 结合能={result.binding_energy:.2f} (>= -3.0，太弱)")
                    if HAS_LOGGER:
                        log_debug("vina", f"Vina对接结果过滤", {
                            "sequence": seq,
                            "reason": f"energy={result.binding_energy:.2f} >= -3.0",
                            "progress": f"{i}/{total_sequences}"
                        })
                else:
                    # 对接失败 - 打印具体原因
                    error_msg = result.error_message if result.error_message else "未知错误"
                    print(f"    【失败】{seq}: {error_msg}")
                    if HAS_LOGGER:
                        log_debug("vina", f"Vina对接失败", {
                            "sequence": seq,
                            "reason": error_msg,
                            "progress": f"{i}/{total_sequences}"
                        })

            except Exception as e:
                print(f"    【异常】{seq}: {e}")
                import traceback
                traceback.print_exc()
                if HAS_LOGGER:
                    log_debug("vina", f"Vina对接异常", {
                        "sequence": seq,
                        "error": str(e),
                        "progress": f"{i}/{total_sequences}"
                    })

            # 【新增】每对接10个分子（或遇到异常后），自动保存一次
            if i % batch_size == 0:
                flush_results(results, is_final=False)

        # 【新增】最终保存所有结果
        flush_results(results, is_final=True)

        # 记录Vina批次总结
        if HAS_LOGGER:
            log_debug("vina", f"Vina批次完成", {
                "total": len(sequences),
                "success": len(results),
                "success_rate": len(results) / len(sequences) if sequences else 0
            })

        return results
    
    def _split_dataset(self, data: List[Tuple[str, float]], 
                       train_ratio: float, val_ratio: float, test_ratio: float):
        """划分数据集"""
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
        """训练EGNN模型"""
        if not HAS_EGNN:
            return False

        try:
            # 保存数据到文件
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

            # 【修改】步骤1：调用 EGNN_1.py 并传入 target_name
            print("  [1/2] 准备EGNN数据 (EGNN_1.py)...")
            result_prep = subprocess.run(
                [
                    sys.executable, "EGNN_1.py",
                    "-s", str(sequences_file),
                    "-e", str(energies_file),
                    "--target", self.target_name  # ← 新增
                ],
                cwd=config.BASE_DIR,
                capture_output=True,
                text=True,
                timeout=3600
            )
            # ... 其余代码保持不变
            
            if result_prep.returncode != 0:
                print(f"  ✗ 数据准备失败: {result_prep.stderr}")
                
                # 记录数据准备失败日志
                if HAS_LOGGER:
                    log_debug("egnn_prep", f"EGNN数据准备失败", {
                        "error": result_prep.stderr,
                        "n_data": len(data)
                    })
                
                return False
            
            print("  ✓ 数据准备完成")
            
            # 【修复】步骤2：再调用 EGNN_23.py 训练模型
            print("  [2/2] 训练EGNN模型 (EGNN_23.py)...")
            result = subprocess.run(
                [sys.executable, "egnn_23.py"],
                cwd=config.BASE_DIR,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                print("  ✓ EGNN训练完成")
                
                # 记录训练成功日志
                if HAS_LOGGER:
                    log_debug("egnn_train", f"EGNN训练完成", {
                        "n_epochs": n_epochs,
                        "n_data": len(data),
                        "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout  # 最后500字符
                    })
                
                return self.load_egnn_model()
            else:
                print(f"  ✗ 训练失败: {result.stderr}")
                
                # 记录训练失败日志
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
        """评估测试集性能（计算R², MAE, RMSE, Pearson r）"""
        if not self.egnn_model or not self.test_set:
            return
        
        sequences = list(self.test_set.keys())
        true_energies = list(self.test_set.values())
        
        pred_energies = self.batch_predict_with_egnn(sequences)
        
        # 计算各项指标
        mae = np.mean([abs(p - t) for p, t in zip(pred_energies, true_energies)])
        rmse = np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(pred_energies, true_energies)]))
        
        # R²
        ss_res = np.sum([(p - t) ** 2 for p, t in zip(pred_energies, true_energies)])
        ss_tot = np.sum([(t - np.mean(true_energies)) ** 2 for t in true_energies])
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Pearson r
        if len(pred_energies) > 1:
            pearson_r = np.corrcoef(pred_energies, true_energies)[0, 1]
        else:
            pearson_r = 0.0
        
        self.test_mae_history.append(mae)
        from EGNN_4 import calculate_pocket_distances, generate_pocket_distance_report

        # 计算距离
        distances = calculate_pocket_distances(sequences, target_name="1LYZ")

        # 生成报告和图表
        stats = generate_pocket_distance_report(distances, output_dir="results/pocket_distances")
        
        print(f"  测试集评估:")
        print(f"    MAE:  {mae:.3f} kcal/mol")
        print(f"    RMSE: {rmse:.3f} kcal/mol")
        print(f"    R²:   {r2:.3f}")
        print(f"    Pearson r: {pearson_r:.3f}")
        
        # 记录到日志
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
    # 核心：EGNN第N轮迭代
    # =================================================================

    def _force_complete(self, partial_sequence: str) -> str:
        """
        直接把所有 'x' 用随机氨基酸填满，走到底

        Args:
            partial_sequence: 部分序列（如 "ACx____C______CG"）

        Returns:
            完整序列（所有 'x' 都被替换）
        """
        import random
        seq_list = list(partial_sequence)
        for i, c in enumerate(seq_list):
            if c in ['x', 'X', '_']:
                seq_list[i] = random.choice(config.ALLOWED_AMINO_ACIDS)
        return ''.join(seq_list)

    def mcts_iteration(self, root: MCTSNode, n_iterations: int) -> MCTSNode:
        """
        执行 n_iterations 次 MCTS 迭代

        每次迭代：
        1. Selection: 从根节点选择到叶节点的路径
        2. Expansion: 如果叶节点未完全扩展，扩展一个子节点
        3. Simulation: 强制走到底，用 EGNN 预测完整序列
        4. Backpropagation: 将奖励沿路径回传
        """
        for i in range(n_iterations):
            # 可选：定期打印进度
            # 每次迭代都打印进度
            print(f"  【MCTS】{i + 1}/{n_iterations}")
            # 1. Selection: 选择路径
            path = self.selector.select_path(
                root,
                can_expand_fn=lambda node: self.expansion_engine.can_expand(node)
            )
            leaf = path[-1]

            # 2. Expansion: 如果非终端且可扩展，扩展一个子节点
            if not leaf.is_terminal and self.expansion_engine.can_expand(leaf):
                new_children = self.expansion_engine.expand(leaf, max_expansions=3)
                if new_children:
                    child = list(new_children.values())[0]
                    path.append(child)
                    leaf = child

            # ============================================================
            # 3. Simulation: 强制走到底！
            # ============================================================
            if leaf.is_terminal:
                # 终端节点：直接用 EGNN 预测
                energy = self.predict_with_egnn(leaf.state.sequence)
            else:
                # 【关键修复】不管走到哪，强制补全为完整序列
                full_seq = self._force_complete(leaf.state.sequence)
                energy = self.predict_with_egnn(full_seq)

            reward = self._energy_to_reward(energy)

            # 4. Backpropagation: 回传奖励
            self.backprop_engine.backpropagate(path, reward)



        return root

    def _energy_to_reward(self, energy: float) -> float:
        """将结合能转换为 [0, 1] 奖励值"""
        # 能量范围假设 -15 ~ 0 kcal/mol
        # -15 最好 → reward=1.0，0 最差 → reward=0.0
        reward = 1.0 - (energy / -15.0)
        return max(0.0, min(1.0, reward))

    def _heuristic_score(self, node: MCTSNode) -> float:
        """启发式分数（用于非终端节点）"""
        seq = node.state.sequence
        # 基于序列完成度
        completed = sum(1 for c in seq if c not in ['_', 'x', 'X'])
        total = len(seq)
        return 0.3 + 0.5 * (completed / total)  # 范围 0.3 ~ 0.8

    def _select_parent_nodes(self, n_nodes: int) -> List[MCTSNode]:
        """
        选择N个母节点

        策略：
        - 第1轮：从根节点扩展19个子节点作为母节点
        - 后续轮次：从候选池中选择表现最好的序列作为母节点
        """
        if self.egnn_round == 1:
            # 第1轮：从根节点创建19个子节点（第一层）
            root = create_root_node()
            expansion_engine = ExpansionEngine(
                egnn_model=self.egnn_model.predict if self.egnn_model else None,
                use_egnn_prior=True
            )

            # 扩展第一层（19个氨基酸）
            children = expansion_engine.expand_level1_amino_acid(root, max_expansions=19)

            # 【关键修复】确保根节点的 visit_count 初始化
            if root.visit_count == 0:
                root.visit_count = 1

            parent_nodes = list(children.values())

            # 【关键修复】记录根节点，供后续回传使用
            self._current_root = root

            return parent_nodes
        else:
            # 后续轮次：从候选池中选择
            if not self.candidate_pool:
                # 候选池为空，重新从根节点开始
                return self._select_parent_nodes_from_root(n_nodes)

            # 按EGNN评分排序（越低越好）
            sorted_candidates = sorted(self.candidate_pool.items(), key=lambda x: x[1])

            # 选择Top-n_nodes作为母节点
            parent_nodes = []
            for seq, score in sorted_candidates[:n_nodes]:
                # 创建节点
                state = PeptideState(sequence=seq, crosslinker=config.CROSSLINKER)
                node = MCTSNode(state=state, prior_prob=1.0)
                parent_nodes.append(node)

            # 【关键修复】创建一个根节点，将所有母节点连接到根
            root = create_root_node()
            if root.visit_count == 0:
                root.visit_count = 1

            for node in parent_nodes:
                node.parent = root
                action = node.decision_action if node.decision_action else node.state.sequence[:10]
                root.children[action] = node

            self._current_root = root

            return parent_nodes
    
    def _select_parent_nodes_from_root(self, n_nodes: int) -> List[MCTSNode]:
        """从根节点重新选择母节点"""
        root = create_root_node()
        expansion_engine = ExpansionEngine(
            egnn_model=self.egnn_model.predict if self.egnn_model else None,
            use_egnn_prior=True
        )
        
        children = expansion_engine.expand_level1_amino_acid(root, max_expansions=n_nodes)
        return list(children.values())


    def extract_top_candidates(self, root: MCTSNode, top_n: int = 100) -> List[Tuple[str, float]]:
        """
        从 MCTS 树中提取 Top-N 终端节点（按 Q 值排序）
        """
        candidates = []

        # DFS 遍历树
        stack = [root]
        visited = set()

        while stack:
            node = stack.pop()
            node_id = id(node)

            if node_id in visited:
                continue
            visited.add(node_id)

            # 如果是终端节点，添加到候选列表
            if node.is_terminal:
                candidates.append((node.state.sequence, node.average_score))

            # 将子节点加入栈
            for child in node.children.values():
                if id(child) not in visited:
                    stack.append(child)

        # 按 Q 值排序（从高到低）
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_n]

    def run(self,
            total_iterations: int = 50000,
            validation_interval: int = 1000,
            validation_top_n: int = 100) -> None:
        """
        运行完整的 MCTS-EGNN 闭环优化

        Args:
            total_iterations: MCTS 总迭代次数
            validation_interval: 每隔多少次迭代进行一次 Vina 验证
            validation_top_n: 每次验证提取的 Top-N 候选数
        """
        print(f"\n开始 MCTS-EGNN 闭环优化")
        print(f"  总迭代次数: {total_iterations}")
        print(f"  验证间隔: {validation_interval}")
        print(f"  每次验证候选数: {validation_top_n}")
        print("=" * 60)

        # 创建根节点
        root = create_root_node()

        # 记录已验证的序列
        validated_sequences = set()

        # 一次性走完所有步
        # 分批执行 MCTS，每 validation_interval 步验证一次
        for iteration in range(validation_interval, total_iterations + 1, validation_interval):
            # 一次走 validation_interval 步
            root = self.mcts_iteration(root, validation_interval)

            # 触发 Vina 验证
            print(f"\n{'=' * 60}")
            print(f"迭代 {iteration}/{total_iterations}: 触发 Vina 验证")
            print(f"{'=' * 60}")

            # 1. 提取 Top-N 候选
            candidates = self.extract_top_candidates(root, validation_top_n)
            print(f"  提取了 {len(candidates)} 个候选序列")

            # 2. 过滤已验证的序列
            new_candidates = [(seq, score) for seq, score in candidates
                              if seq not in validated_sequences]

            if not new_candidates:
                print("  没有新的候选序列，跳过验证")
                continue

            print(f"  其中 {len(new_candidates)} 个是新的")

            # 3. Vina 验证
            vina_results = []
            for seq, mcts_score in new_candidates[:validation_top_n]:
                energy = self._vina_dock(seq)
                if energy is not None and energy < 0:
                    vina_results.append((seq, energy))
                    validated_sequences.add(seq)

            print(f"  Vina 验证成功: {len(vina_results)} 个")

            if vina_results:
                self._update_training_data(vina_results)
                self._finetune_egnn()
                self._evaluate_test_set()

        print(f"\n{'=' * 60}")
        print("MCTS-EGNN 闭环优化完成！")
        print(f"  总迭代次数: {total_iterations}")
        print(f"  已验证序列数: {len(validated_sequences)}")
        print(f"  候选池大小: {len(self.candidate_pool)}")
        print(f"{'=' * 60}")


    def _vina_dock(self, sequence: str) -> Optional[float]:
        """调用 Vina 对接单个序列"""
        try:
            from vina import vina_dock
            energy = vina_dock(
                sequence=sequence,
                target_name=self.target_name,
                verbose=False
            )
            return energy
        except Exception as e:
            print(f"  Vina 对接失败 {sequence[:20]}...: {e}")
            return None

    def _update_training_data(self, vina_results: List[Tuple[str, float]]):
        """更新训练数据"""
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
        """微调 EGNN 模型"""
        if not self.vina_validated:
            print("  没有新数据，跳过微调")
            return

        print(f"  微调 EGNN（使用 {len(self.vina_validated)} 个数据，{n_epochs} epochs）...")
        data = list(self.vina_validated.items())
        self._train_egnn(data, n_epochs=n_epochs)

    
    def _select_and_validate_filled_sequences(self, 
                                               filled_sequences: List[Tuple[str, float]], 
                                               n_validate: int) -> List[Tuple[str, float]]:
        """
        从随机填充的完整序列中选择Top-h(N)个进行Vina验证
        
        Args:
            filled_sequences: [(sequence, egnn_energy), ...] 随机填充的完整序列
            n_validate: 要验证的数量
        
        Returns:
            [(sequence, vina_energy), ...] Vina验证结果
        """
        if not filled_sequences:
            print("  【警告】没有随机填充的完整序列")
            return []
        
        # 去重：同一序列保留EGNN预测能量最好的
        unique_sequences = {}
        for seq, energy in filled_sequences:
            if seq not in unique_sequences or energy < unique_sequences[seq]:
                unique_sequences[seq] = energy
        
        # 过滤掉已验证的序列
        unvalidated = {seq: energy for seq, energy in unique_sequences.items() 
                       if seq not in self.vina_validated}
        
        if not unvalidated:
            print(f"  【警告】所有随机填充序列都已验证过")
            print(f"    总序列数: {len(unique_sequences)}")
            print(f"    已验证: {len(self.vina_validated)}")
            return []
        
        # 按EGNN预测能量排序（越低越好）
        sorted_sequences = sorted(unvalidated.items(), key=lambda x: x[1])
        
        # 选择Top-n_validate
        to_validate = sorted_sequences[:n_validate]
        sequences = [seq for seq, _ in to_validate]
        
        print(f"  从 {len(filled_sequences)} 个填充序列中")
        print(f"  去重后: {len(unique_sequences)} 个")
        print(f"  未验证: {len(unvalidated)} 个")
        print(f"  选择Top-{len(sequences)}个进行Vina验证")
        
        # 运行Vina
        return self._run_vina_batch(sequences)
    
    def _update_egnn_with_new_data(self, vina_results: List[Tuple[str, float]]) -> bool:
        """
        使用新的Vina数据更新EGNN
        
        1. 8:1:1划分新数据
        2. 与历史数据合并
        3. 微调EGNN
        4. 评估测试集
        """
        if not vina_results:
            print("  【错误】Vina对接没有返回任何有效数据！")
            print("  可能原因：")
            print("    1. Vina对接全部失败")
            print("    2. 所有对接结果的结合能 >= 0（被过滤）")
            print("    3. 候选池中没有未验证的终端节点")
            return False
        
        # 更新已验证集合
        for seq, energy in vina_results:
            self.vina_validated[seq] = energy
        
        # 8:1:1划分
        train_data, val_data, test_data = self._split_dataset(
            vina_results, 0.8, 0.1, 0.1
        )
        
        print(f"  划分: 训练{len(train_data)} / 验证{len(val_data)} / 测试{len(test_data)}")
        
        # 更新测试集
        for seq, energy in test_data:
            self.test_set[seq] = energy
        
        # 限制测试集大小（保留最新的100个）
        if len(self.test_set) > 100:
            items = list(self.test_set.items())
            self.test_set = dict(items[-100:])
        
        # 合并所有历史数据用于训练
        all_train_data = list(self.vina_validated.items())
        
        # 微调EGNN（较少epoch）
        print(f"  微调EGNN（使用{len(all_train_data)}个数据）...")
        return self._train_egnn(all_train_data, n_epochs=20)

    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'egnn_round': self.egnn_round,
            'candidate_pool': self.candidate_pool,
            'vina_validated': self.vina_validated,
            'test_set': self.test_set,
            'test_mae_history': self.test_mae_history,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.results_dir / f"checkpoint_round{self.egnn_round}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"  ✓ 检查点保存: {checkpoint_path}")


# =================================================================
# 主程序入口
# =================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自适应MCTS-EGNN闭环优化 V3")
    parser.add_argument("--max-iterations", type=int, default=50000, help="MCTS总迭代次数")
    parser.add_argument("--validation-interval", type=int, default=1000, help="验证间隔")
    parser.add_argument("--validation-top-n", type=int, default=100, help="每次验证提取的候选数")
    parser.add_argument("-t", "--target", required=True, help="靶点名称（如1LYZ）")

    parser.add_argument("--cold-start-n", type=int, default=1500, help="冷启动序列数")
    
    args = parser.parse_args()
    
    engine = AdaptiveMCTSEngineV3(args.target)
    
    # 检查EGNN模型
    if engine.egnn_model_path.exists():
        print(f"\n检测到已有EGNN模型，跳过冷启动")
        engine.load_egnn_model()

    else:
        print(f"\n未检测到EGNN模型，执行冷启动...")
        if not engine.cold_start(n_sequences=args.cold_start_n):
            print("冷启动失败！")
            sys.exit(1)

    engine.run(
        total_iterations=args.max_iterations,  # 需要添加到 argparse
        validation_interval=args.validation_interval,  # 需要添加到 argparse
        validation_top_n=args.validation_top_n  # 需要添加到 argparse
    )
    

    # 关闭日志
    if HAS_LOGGER:
        close_logger()


if __name__ == "__main__":
    main()

#