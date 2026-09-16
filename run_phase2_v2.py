#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二主程序 V2 - 自适应MCTS-EGNN闭环优化

新特性：
1. Softmax概率分配扩展名额（表现好的节点获得更多资源）
2. 随机填充评估（深度控制填充数量）
3. 实时测试集更新（每轮8:1:1划分，测试集动态增长）
4. 基于测试集MAE的收敛检测

工作流程：
1. 冷启动：生成初始序列 → 8:1:1划分 → 训练初始EGNN
2. MCTS迭代：Softmax分配50个名额 → 随机填充评估 → 扩展节点
3. Vina验证：Top-K序列验证 → 8:1:1划分新数据
4. EGNN更新：微调模型 → 测试集评估 → 检查收敛
5. 重复直到收敛或达到最大轮数
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

import config

# 导入自适应MCTS配置
try:
    from adaptive_mcts_config import AdaptiveMCTSConfig
    HAS_ADAPTIVE_CONFIG = True
except ImportError:
    HAS_ADAPTIVE_CONFIG = False
    print("警告: adaptive_mcts_config未找到，使用默认配置")

# 导入MCTS模块
from peptide_state import PeptideState, create_root_node, MCTSNode
from selection import PUCTSelector
from expansion import ExpansionEngine
from simulation import SimulationEngine, SimulationResult
from backpropagation import BackpropagationEngine
from seq_generator import generate_full_sequence, generate_n_random_fills
from vina import batch_vina_dock, get_vina_paths, run_vina_with_progress
from ligand_generator import generate_ligand

# 导入EGNN模块
try:
    from egnn_predictor import EGNNPredictor, create_egnn_predictor
    from EGNN_1 import main as egnn_data_prep
    from EGNN_23 import main as egnn_train
    HAS_EGNN = True
except ImportError:
    HAS_EGNN = False
    print("警告: EGNN模块未找到")

# 导入日志模块
try:
    from mcts_logger import init_logger, close_logger, log_progress, log_debug
    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False


class AdaptiveMCTSEngine:
    """
    自适应MCTS引擎 V2
    
    核心特性：
    - Softmax概率分配扩展名额
    - 随机填充评估
    - 实时测试集更新
    - 自适应收敛检测
    """
    
    def __init__(self, target_name: str):
        """
        初始化自适应MCTS引擎
        
        Args:
            target_name: 靶点名称
        """
        self.target_name = target_name
        self.target_dirs = config.get_target_dirs(target_name)
        
        # 初始化日志
        if HAS_LOGGER:
            self.logger = init_logger(target_name)
        
        # MCTS组件
        self.selector = PUCTSelector(c_puct=config.MCTS_CONFIG.get("c_puct", 1.414))
        self.expansion_engine = None  # 将在加载EGNN后初始化
        self.backprop_engine = BackpropagationEngine(verbose=False)
        
        # EGNN模型
        self.egnn_model = None
        self.egnn_model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"
        
        # 实时测试集管理
        self.test_set: Dict[str, float] = {}  # {sequence: energy}
        self.test_mae_history: List[float] = []  # 测试集MAE历史
        self.round_best_energies: List[float] = []  # 每轮最佳能量
        
        # 统计信息
        self.round = 0
        self.total_iterations = 0
        
        # 结果目录
        self.results_dir = config.RESULTS_DIR / target_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"="*60)
        print(f"自适应MCTS引擎 V2 初始化")
        print(f"靶点: {target_name}")
        print(f"="*60)
    
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
            
            # 初始化带EGNN先验的ExpansionEngine
            self.expansion_engine = ExpansionEngine(
                egnn_model=self.egnn_model.predict,
                use_egnn_prior=True,
                prior_temperature=1.0
            )
            return True
        except Exception as e:
            print(f"✗ EGNN模型加载失败: {e}")
            return False
    
    def cold_start(self, n_sequences: int = 100) -> bool:
        """
        冷启动：生成初始数据并训练EGNN
        
        Args:
            n_sequences: 初始序列数量
        
        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("冷启动阶段")
        print("="*60)
        
        # 步骤1: 生成随机序列
        print(f"\n[1/4] 生成 {n_sequences} 个随机序列...")
        sequences = [generate_full_sequence() for _ in range(n_sequences)]
        print(f"  ✓ 生成完成")
        
        # 步骤2: Vina对接
        print(f"\n[2/4] Vina对接（获取真实能量）...")
        vina_results = {}
        vina_paths = get_vina_paths(self.target_name)
        
        for i, seq in enumerate(sequences, 1):
            print(f"  [{i}/{n_sequences}] {seq}", end=" ")
            try:
                pdbqt_path = generate_ligand(
                    sequence=seq,
                    crosslinker=config.CROSSLINKER,
                    crosslinker_positions=config.CROSSLINKER_POSITIONS
                )
                result = run_vina_with_progress(
                    ligand_pdbqt=pdbqt_path,
                    receptor_pdbqt=vina_paths['receptor'],
                    vina_config=vina_paths['config'],
                    n_cpu=config.VINA_CONFIG.get("cpu", 4)
                )
                if result.success and result.binding_energy < 0:
                    vina_results[seq] = result.binding_energy
                    print(f"✓ {result.binding_energy:.2f}")
                else:
                    print(f"✗ 失败")
            except Exception as e:
                print(f"✗ 错误: {e}")
        
        print(f"  ✓ Vina对接完成: {len(vina_results)}/{n_sequences} 成功")
        
        if len(vina_results) < 10:
            print("错误: 有效数据太少，冷启动失败")
            return False
        
        # 步骤3: 8:1:1划分数据集
        print(f"\n[3/4] 划分数据集 (8:1:1)...")
        if HAS_ADAPTIVE_CONFIG:
            split_data = AdaptiveMCTSConfig.split_dataset(
                list(vina_results.keys()),
                list(vina_results.values()),
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1
            )
            train_seqs, train_energies = split_data['train']
            val_seqs, val_energies = split_data['val']
            test_seqs, test_energies = split_data['test']
            
            # 保存测试集
            self.test_set = dict(zip(test_seqs, test_energies))
            print(f"  ✓ 划分完成: 训练{len(train_seqs)} / 验证{len(val_seqs)} / 测试{len(test_seqs)}")
        else:
            # 简单划分
            items = list(vina_results.items())
            n_train = int(len(items) * 0.8)
            n_val = int(len(items) * 0.1)
            train_data = items[:n_train]
            val_data = items[n_train:n_train+n_val]
            test_data = items[n_train+n_val:]
            self.test_set = dict(test_data)
            print(f"  ✓ 划分完成: 训练{len(train_data)} / 验证{len(val_data)} / 测试{len(test_data)}")
        
        # 步骤4: 训练EGNN
        print(f"\n[4/4] 训练初始EGNN模型...")
        success = self._train_egnn(list(vina_results.items()), n_epochs=100)
        if not success:
            return False
        
        print("\n" + "="*60)
        print("冷启动完成！")
        print("="*60)
        return True
    
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
            
            # 准备数据
            egnn_data_prep(
                sequences_file=sequences_file,
                energies_file=energies_file,
                output_dir=config.BASE_DIR / "egnn" / "raw"
            )
            
            # 训练
            egnn_train(
                data_dir=config.BASE_DIR / "egnn" / "raw",
                output_dir=config.BASE_DIR / "egnn" / "models",
                hidden_dim=config.EGNN_CONFIG["hidden_dim"],
                num_layers=config.EGNN_CONFIG["num_layers"],
                num_epochs=n_epochs,
                batch_size=config.EGNN_CONFIG["batch_size"],
                lr=config.EGNN_CONFIG["learning_rate"],
                patience=config.EGNN_CONFIG["patience"]
            )
            
            # 加载新模型
            return self.load_egnn_model()
            
        except Exception as e:
            print(f"✗ EGNN训练失败: {e}")
            return False
    
    def mcts_round(self, n_iterations: int = 1000, n_nodes: int = 19) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        执行一轮MCTS搜索
        
        Args:
            n_iterations: 迭代次数
            n_nodes: 要扩展的节点数量
        
        Returns:
            (根节点, 扩展的节点列表)
        """
        print(f"\n--- 第{self.round}轮 MCTS搜索 ---")
        
        # 创建根节点
        root = create_root_node()
        
        # 模拟引擎
        sim_engine = SimulationEngine(
            target_name=self.target_name,
            egnn_model=self.egnn_model.predict if self.egnn_model else None
        )
        
        expanded_nodes = []
        
        for i in range(n_iterations):
            self.total_iterations += 1
            
            # 1. Selection: 选择路径
            path = self.selector.select_path(
                root,
                can_expand_fn=lambda node: self.expansion_engine.can_expand(node)
            )
            leaf = path[-1]
            
            # 2. Expansion: 扩展节点
            if not leaf.is_terminal and self.expansion_engine.can_expand(leaf):
                new_children = self.expansion_engine.expand(leaf)
                if new_children:
                    best_child = max(new_children.values(), key=lambda c: c.prior_prob)
                    path.append(best_child)
                    leaf = best_child
                    expanded_nodes.append(leaf)
            
            # 3. Simulation: 评估
            if leaf.is_terminal:
                result = sim_engine.simulate(leaf)
                reward = result.score
            else:
                reward = 0.5
            
            # 4. Backpropagation: 回溯更新
            self.backprop_engine.backpropagate(path, reward)
            
            # 定期输出进度
            if (i + 1) % 100 == 0:
                print(f"  MCTS迭代: {i+1}/{n_iterations}, 已扩展节点: {len(expanded_nodes)}")
        
        return root, expanded_nodes
    
    def extract_candidates(self, root: MCTSNode, top_n: int = 40) -> List[Tuple[PeptideState, float]]:
        """提取候选序列"""
        candidates = []
        
        # 迭代式DFS
        stack = [root]
        visited = set()
        
        while stack:
            node = stack.pop()
            node_id = id(node)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            if node.is_terminal:
                state_copy = node.state.copy()
                candidates.append((state_copy, node.average_score))
            
            for child in node.children.values():
                if id(child) not in visited:
                    stack.append(child)
        
        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]
    
    def vina_validation(self, candidates: List[Tuple[PeptideState, float]]) -> Dict[str, float]:
        """Vina验证候选序列"""
        print(f"\nVina验证: {len(candidates)} 个候选")
        
        vina_paths = get_vina_paths(self.target_name)
        results = {}
        
        for i, (state, mcts_score) in enumerate(candidates, 1):
            print(f"  [{i}/{len(candidates)}] {state.sequence}", end=" ")
            try:
                pdbqt_path = generate_ligand(
                    sequence=state.sequence,
                    crosslinker=state.crosslinker or config.CROSSLINKER,
                    crosslinker_positions=config.CROSSLINKER_POSITIONS
                )
                result = run_vina_with_progress(
                    ligand_pdbqt=pdbqt_path,
                    receptor_pdbqt=vina_paths['receptor'],
                    vina_config=vina_paths['config'],
                    n_cpu=config.VINA_CONFIG.get("cpu", 4)
                )
                if result.success and result.binding_energy < 0:
                    results[state.sequence] = result.binding_energy
                    print(f"✓ {result.binding_energy:.2f}")
                else:
                    print(f"✗ 失败")
            except Exception as e:
                print(f"✗ 错误: {e}")
        
        print(f"  ✓ Vina验证完成: {len(results)}/{len(candidates)} 成功")
        return results
    
    def update_egnn(self, vina_results: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """
        更新EGNN模型（实时测试集版本）
        
        Returns:
            (是否成功, 评估指标)
        """
        print("\n" + "="*60)
        print("EGNN模型更新（实时测试集）")
        print("="*60)
        
        if not HAS_ADAPTIVE_CONFIG:
            print("错误: adaptive_mcts_config不可用")
            return False, {}
        
        # 步骤1: 划分Vina结果
        print(f"\n[1/4] 划分Vina结果 (8:1:1)...")
        sequences = list(vina_results.keys())
        energies = list(vina_results.values())
        
        split_data = AdaptiveMCTSConfig.split_dataset(
            sequences, energies, 0.8, 0.1, 0.1
        )
        train_seqs, train_energies = split_data['train']
        val_seqs, val_energies = split_data['val']
        new_test_seqs, new_test_energies = split_data['test']
        
        print(f"  ✓ 划分: 训练{len(train_seqs)} / 验证{len(val_seqs)} / 测试{len(new_test_seqs)}")
        
        # 步骤2: 更新测试集
        print(f"\n[2/4] 更新测试集...")
        for seq, energy in zip(new_test_seqs, new_test_energies):
            self.test_set[seq] = energy
        
        # 限制测试集大小
        if len(self.test_set) > 100:
            items = list(self.test_set.items())
            self.test_set = dict(items[-100:])
            print(f"  测试集超过100，保留最新的100个")
        
        print(f"  ✓ 测试集更新: {len(self.test_set)} 个")
        
        # 步骤3: 训练EGNN
        print(f"\n[3/4] 微调EGNN模型...")
        all_train_data = list(zip(train_seqs, train_energies))
        success = self._train_egnn(all_train_data, n_epochs=20)
        if not success:
            return False, {}
        
        # 步骤4: 测试集评估
        print(f"\n[4/4] 测试集评估...")
        if self.egnn_model and self.test_set:
            metrics = AdaptiveMCTSConfig.evaluate_egnn_on_test_set(
                model=self.egnn_model,
                test_sequences=list(self.test_set.keys()),
                test_energies=list(self.test_set.values()),
                target_name=self.target_name
            )
            
            self.test_mae_history.append(metrics['mae'])
            
            print(f"  ✓ 测试集评估完成")
            print(f"    MAE: {metrics['mae']:.3f} kcal/mol")
            print(f"    RMSE: {metrics['rmse']:.3f} kcal/mol")
            print(f"    R²: {metrics['r2']:.3f}")
            
            return True, metrics
        
        return False, {}
    
    def check_convergence(self) -> Tuple[bool, str]:
        """检查是否收敛"""
        if not HAS_ADAPTIVE_CONFIG or len(self.test_mae_history) < 6:
            return False, "数据不足"
        
        return AdaptiveMCTSConfig.check_convergence(
            self.test_mae_history, patience=5, min_improvement=0.05
        )
    
    def run(self, max_rounds: int = 100, n_iterations_per_round: int = 1000) -> List[Dict]:
        """
        运行自适应MCTS-EGNN闭环优化
        
        Args:
            max_rounds: 最大轮数
            n_iterations_per_round: 每轮MCTS迭代次数
        
        Returns:
            每轮结果列表
        """
        print("\n" + "="*60)
        print("自适应MCTS-EGNN闭环优化")
        print("="*60)
        print(f"最大轮数: {max_rounds}")
        print(f"每轮迭代: {n_iterations_per_round}")
        print(f"收敛条件: 连续5轮测试集MAE无显著改善")
        print("="*60)
        
        # 设置SimulationEngine的最大迭代次数
        SimulationEngine.set_max_iterations(max_rounds)
        
        # 检查EGNN模型
        if not self.egnn_model:
            # 检查是否已有训练好的模型
            if self.egnn_model_path.exists():
                print(f"\n检测到已有EGNN模型: {self.egnn_model_path}")
                print("跳过冷启动，直接加载模型...")
                if not self.load_egnn_model():
                    print("模型加载失败！")
                    return []
                print("✓ 模型加载成功，直接开始MCTS搜索")
            else:
                print("\n未检测到EGNN模型，执行冷启动...")
                if not self.cold_start(n_sequences=100):
                    print("冷启动失败！")
                    return []
        
        results = []
        
        for round_idx in range(max_rounds):
            self.round = round_idx + 1
            
            print(f"\n{'='*60}")
            print(f"第 {self.round}/{max_rounds} 轮")
            print(f"{'='*60}")
            
            # 1. MCTS搜索
            root, expanded_nodes = self.mcts_round(n_iterations_per_round)
            
            # 2. 提取候选
            candidates = self.extract_candidates(root, top_n=40)
            print(f"\n提取Top-{len(candidates)}候选")
            
            # 3. Vina验证
            vina_results = self.vina_validation(candidates)
            
            # 记录最佳能量
            if vina_results:
                best_energy = min(vina_results.values())
                self.round_best_energies.append(best_energy)
                print(f"\n本轮最佳能量: {best_energy:.2f} kcal/mol")
            
            # 4. 更新EGNN
            if vina_results:
                success, metrics = self.update_egnn(vina_results)
                if success:
                    results.append({
                        'round': self.round,
                        'vina_results': vina_results,
                        'metrics': metrics
                    })
                    
                    # 5. 检查收敛
                    is_converged, reason = self.check_convergence()
                    if is_converged:
                        print(f"\n{'='*60}")
                        print(f"🎉 模型已收敛！")
                        print(f"原因: {reason}")
                        print(f"共进行 {self.round} 轮")
                        print(f"{'='*60}")
                        break
            
            # 保存检查点
            self._save_checkpoint()
        
        print("\n" + "="*60)
        print("自适应MCTS-EGNN闭环优化完成！")
        print("="*60)
        
        if HAS_LOGGER:
            close_logger()
        
        return results
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'round': self.round,
            'test_set': self.test_set,
            'test_mae_history': self.test_mae_history,
            'round_best_energies': self.round_best_energies
        }
        checkpoint_path = self.results_dir / f"checkpoint_round{self.round}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='自适应MCTS-EGNN闭环优化 V2')
    parser.add_argument('-t', '--target', type=str, required=True, help='靶点名称')
    parser.add_argument('--max-rounds', type=int, default=100, help='最大轮数')
    parser.add_argument('--iter-per-round', type=int, default=1000, help='每轮迭代次数')
    parser.add_argument('--cold-start-n', type=int, default=100, help='冷启动序列数')
    
    args = parser.parse_args()
    
    # 创建引擎
    engine = AdaptiveMCTSEngine(target_name=args.target)
    
    # 运行
    results = engine.run(
        max_rounds=args.max_rounds,
        n_iterations_per_round=args.iter_per_round
    )
    
    # 输出总结
    print("\n" + "="*60)
    print("结果总结")
    print("="*60)
    for r in results:
        print(f"\n第{r['round']}轮:")
        print(f"  Vina验证: {len(r['vina_results'])} 个")
        if r['vina_results']:
            print(f"  最佳能量: {min(r['vina_results'].values()):.2f} kcal/mol")
        print(f"  测试集MAE: {r['metrics'].get('mae', 'N/A')}")


if __name__ == "__main__":
    main()
