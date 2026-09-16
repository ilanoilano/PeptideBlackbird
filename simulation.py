#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Simulation模块 V2 (simulation.py)
支持三层决策的模拟引擎

关键改进：
- 使用MCTS确定的交联剂/拓扑，不再随机选择
- 每个状态（序列+交联剂+二硫键）对应唯一的分子构象
- 【修复】使用项目目录下的临时文件夹，避免硬编码 /tmp
"""

import sys
import random
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import config
from peptide_state import MCTSNode, PeptideState
from ligand_generator import generate_ligand
from egnn_predictor import create_egnn_predictor, EGNNPredictor
from seq_generator import generate_n_random_fills

# 导入MCTS日志模块
try:
    from mcts_logger import get_logger, log_progress, log_debug
except ImportError:
    get_logger = None
    log_progress = None
    log_debug = None


# 【修复】使用项目目录下的临时文件夹，避免硬编码 /tmp
TEMP_DIR = config.BASE_DIR / "temp" / "mcts_simulation"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SimulationResult:
    """模拟结果"""
    score: float           # 归一化分数 [0, 1]
    raw_energy: float      # 原始结合能（kcal/mol）
    details: Dict          # 详细信息


class SimulationEngine:
    """
    MCTS模拟引擎 V2
    
    使用MCTS确定的完整状态（序列+交联剂+二硫键）
    """
    
    # 类级别的计数器，用于跟踪生成的分子数量
    _molecule_counter = 0
    _egnn_iteration = 0
    _max_iterations = 100  # 默认最大迭代次数
    
    def __init__(self,
                 target_name: str,
                 egnn_model: Optional[Callable] = None,
                 n_simulations: int = 1):  # V2中n_simulations主要用于不同构象
        """
        初始化模拟引擎
        
        Args:
            target_name: 靶点名称
            egnn_model: EGNN预测函数
            n_simulations: 每个状态的模拟次数（不同随机种子）
        """
        self.target_name = target_name
        self.egnn_model = egnn_model
        self.n_simulations = n_simulations
        
        # 【修复】使用项目目录下的临时文件夹
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)
        
        # 【新增】PDBQT 缓存，避免重复生成分子
        self._pdbqt_cache = {}  # {sequence: pdbqt_path}
    
    @classmethod
    def increment_molecule_counter(cls):
        """增加分子计数器"""
        cls._molecule_counter += 1
        return cls._molecule_counter
    
    @classmethod
    def set_egnn_iteration(cls, iteration: int):
        """设置当前EGNN迭代轮次"""
        cls._egnn_iteration = iteration
    
    @classmethod
    def set_max_iterations(cls, max_iterations: int):
        """设置最大迭代次数"""
        cls._max_iterations = max_iterations
    
    @classmethod
    def get_stats(cls):
        """获取统计信息"""
        return {
            "molecule_count": cls._molecule_counter,
            "egnn_iteration": cls._egnn_iteration,
            "max_iterations": cls._max_iterations
        }
    
    def get_crosslinker_config(self, state: PeptideState) -> tuple:
        """
        从状态获取交联剂配置
        
        Returns:
            (crosslinker_type, crosslinker_positions)
        """
        crosslinker = state.crosslinker
        
        if crosslinker is None:
            # 线性肽，无交联
            return None, []
        
        elif crosslinker == "disulfide":
            # 二硫键：使用状态中的配对
            if state.disulfide_bonds:
                # 返回第一个配对（简化处理）
                bond = state.disulfide_bonds[0]
                return "disulfide", list(bond)
            else:
                return None, []
        
        else:
            # 化学交联剂：TBMB/TATA/TBAB
            # 使用配置中的位置，或自动检测Cys
            cys_positions = state.get_cys_positions()
            
            if crosslinker in ["TBMB", "TATA"]:
                # 需要3个Cys
                positions = cys_positions[:3] if len(cys_positions) >= 3 else cys_positions
            elif crosslinker == "TBAB":
                # 需要4个Cys
                positions = cys_positions[:4] if len(cys_positions) >= 4 else cys_positions
            else:
                positions = []
            
            return crosslinker, positions
    
    def predict_with_egnn(self, state: PeptideState, seed: int = 42, 
                          mcts_depth: int = 0) -> float:
        """
        使用EGNN预测结合能
        
        Args:
            state: 肽状态
            seed: 随机种子（用于生成不同构象）
            mcts_depth: MCTS当前深度
        
        Returns:
            预测结合能（kcal/mol）
        """
        # 获取交联剂配置
        crosslinker, positions = self.get_crosslinker_config(state)
        
        # 记录开始时间
        start_time = time.time()
        
        # 增加分子计数
        mol_count = self.increment_molecule_counter()
        
        try:
            # 生成分子
            # 【新增】检查缓存
            cache_key = f"{state.sequence}_{seed}"
            if cache_key in self._pdbqt_cache:
                pdbqt_path = self._pdbqt_cache[cache_key]
                print(f"  【缓存】使用已生成的配体: {pdbqt_path.name}")
            else:
                pdbqt_path = generate_ligand(
                    sequence=state.sequence,
                    crosslinker=crosslinker,
                    crosslinker_positions=positions if positions else None,
                    output_dir=self.temp_dir,
                    random_seed=seed
                )
                # 缓存生成的 PDBQT 路径
                self._pdbqt_cache[cache_key] = pdbqt_path
            
            # EGNN预测
            if self.egnn_model is None:
                # 懒加载EGNN模型
                self.egnn_model = create_egnn_predictor()
            
            energy = self.egnn_model.predict(pdbqt_path)
            
            # 输出进度信息
            if log_progress:
                log_progress(
                    egnn_iter=self._egnn_iteration,
                    mcts_depth=mcts_depth,
                    random_count=mol_count,
                    sequence=state.sequence,
                    energy=energy,
                    max_iterations=self._max_iterations
                )
            
            # 记录调试信息
            if log_debug:
                log_debug("simulation", "EGNN预测完成", {
                    "sequence": state.sequence,
                    "crosslinker": crosslinker,
                    "energy": energy,
                    "mcts_depth": mcts_depth,
                    "generation_time": round(time.time() - start_time, 3)
                })
            
            return energy
            
        except Exception as e:
            error_msg = f"EGNN预测失败: {e}"
            if log_debug:
                log_debug("simulation", error_msg, {
                    "sequence": state.sequence,
                    "crosslinker": crosslinker,
                    "error": str(e)
                })
            # 返回一个较差的分数作为惩罚
            return 0.0  # 0表示无效/失败
    
    def _get_completion_count(self, depth: int) -> int:
        """
        【新增】根据当前深度计算随机补全数量N

        公式: N = max(MIN_RANDOM_FILL, MAX_RANDOM_FILL - depth * FILL_DECREMENT_PER_DEPTH)

        Args:
            depth: 当前节点深度（已填充的氨基酸数）

        Returns:
            需要生成的随机序列数量
        """
        max_fill = config.MAX_RANDOM_FILL
        min_fill = config.MIN_RANDOM_FILL
        decrement = config.FILL_DECREMENT_PER_DEPTH

        n = max_fill - depth * decrement
        return max(min_fill, n)

    def _random_complete(self, state: PeptideState, n: int) -> List[str]:
        """
        【新增】对状态进行随机补全，生成N个完整序列

        Args:
            state: 当前肽状态（含占位符）
            n: 需要生成的序列数量

        Returns:
            N个完整序列列表
        """
        return generate_n_random_fills(state.sequence, n)

    def calculate_heuristic_score(self, state: PeptideState) -> float:
        """
        计算启发式分数（用于非终端节点）

        基于：
        - 序列完成度
        - Cys数量（用于交联）
        - 氨基酸组成
        """
        score = 0.0

        # 序列完成度
        completed = sum(1 for aa in state.sequence if aa != '_')
        total = len(state.sequence)
        score += 0.3 * (completed / total)

        # Cys数量（用于交联）
        cys_count = len(state.get_cys_positions())
        if state.crosslinker in ["TBMB", "TATA"]:
            # 需要至少3个Cys
            score += 0.3 * min(cys_count / 3, 1.0)
        elif state.crosslinker == "TBAB":
            # 需要至少4个Cys
            score += 0.3 * min(cys_count / 4, 1.0)
        elif state.crosslinker == "disulfide":
            # 需要至少2个Cys
            score += 0.3 * min(cys_count / 2, 1.0)

        # 多样性奖励（不同氨基酸类型）
        aa_types = set(state.sequence.replace('_', ''))
        score += 0.1 * min(len(aa_types) / 10, 1.0)

        return score
    
    def simulate(self, node: MCTSNode, verbose: bool = False) -> SimulationResult:
        """
        执行模拟（rollout）

        【修改】新流程：
        1. 如果非终端状态，进行随机补全生成N个序列
        2. EGNN预测N个序列的结合能
        3. 计算平均值作为Q值

        Args:
            node: MCTS节点
            verbose: 是否打印详细信息

        Returns:
            SimulationResult
        """
        state = node.state

        if verbose:
            print(f"\n[Simulation] 开始模拟")
            print(f"  序列: {state.sequence}")
            print(f"  交联剂: {state.crosslinker}")

        # 【修改】计算当前深度和补全数量
        depth = sum(1 for c in state.sequence if c not in ['_', 'x', 'X'])
        n_completions = self._get_completion_count(depth)

        if verbose:
            print(f"  当前深度: {depth}, 补全数量: {n_completions}")

        # 【修改】生成N个随机补全序列
        completed_sequences = self._random_complete(state, n_completions)

        if verbose:
            print(f"  生成 {len(completed_sequences)} 个随机补全序列")

        # EGNN预测所有补全序列
        energies = []
        for seq in completed_sequences:
            # 创建临时状态
            temp_state = PeptideState(
                sequence=seq,
                crosslinker=state.crosslinker,
                disulfide_bonds=state.disulfide_bonds
            )
            energy = self.predict_with_egnn(temp_state, random.randint(1, 10000), mcts_depth=depth)
            energies.append(energy)

        # 计算平均能量作为得分
        if energies:
            avg_energy = sum(energies) / len(energies)
            # 归一化到[0, 1]，能量越低越好
            score = max(0.0, min(1.0, 1.0 - (avg_energy / -15.0)))
        else:
            avg_energy = 0.0
            score = 0.0

        if verbose:
            print(f"  平均能量: {avg_energy:.2f}, 得分: {score:.4f}")

        return SimulationResult(
            score=score,
            raw_energy=avg_energy,
            details={
                "type": "random_complete",
                "n_completions": n_completions,
                "energies": energies,
                "avg_energy": avg_energy
            }
        )


def main():
    """测试代码"""
    print("="*60)
    print("SimulationEngine测试")
    print("="*60)
    
    # 创建引擎
    engine = SimulationEngine(
        target_name="1LYZ",
        egnn_model=None,  # 使用默认EGNN
        n_simulations=3
    )
    
    # 测试1：完整状态（TBMB交联）
    print("\n[测试1] 完整状态（TBMB交联）")
    state1 = PeptideState(
        sequence="ACARNDCMVFLWPCG",
        crosslinker="TBMB",
        disulfide_bonds=[]
    )
    node1 = MCTSNode(state=state1)
    
    result1 = engine.simulate(node1, verbose=True)
    print(f"结果: score={result1.score:.4f}, energy={result1.raw_energy:.4f}")
    
    # 测试2：完整状态（二硫键）
    print("\n[测试2] 完整状态（二硫键）")
    state2 = PeptideState(
        sequence="ACARNDCMVFLWPCG",
        crosslinker="disulfide",
        disulfide_bonds=[(1, 8)]
    )
    node2 = MCTSNode(state=state2)
    
    result2 = engine.simulate(node2, verbose=True)
    print(f"结果: score={result2.score:.4f}, energy={result2.raw_energy:.4f}")
    
    # 测试3：不完整状态
    print("\n[测试3] 不完整状态")
    state3 = PeptideState(
        sequence="ACARNDC______CG",
        crosslinker=None,
        disulfide_bonds=[]
    )
    node3 = MCTSNode(state=state3)
    
    result3 = engine.simulate(node3, verbose=True)
    print(f"结果: score={result3.score:.4f} (启发式)")
    
    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)


if __name__ == "__main__":
    main()
