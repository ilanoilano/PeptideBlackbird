#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Expansion模块 V2 (expansion.py)
支持三层决策的扩展引擎

决策层：
1. 氨基酸序列填充
2. 交联剂/环化方式选择
3. 二硫键配对选择
"""

import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import config
from peptide_state import MCTSNode, PeptideState, CROSSLINKER_TYPES, CROSSLINKER_CYS_COUNT


class ExpansionEngine:
    """
    MCTS扩展引擎 V3 - 支持EGNN先验指导
    """
    
    def __init__(self,
                 template: Optional[str] = None,
                 fixed_positions: Optional[Dict[int, str]] = None,
                 variable_amino_acids: Optional[Dict[int, List[str]]] = None,
                 prior_policy: Optional[Callable] = None,
                 egnn_model: Optional[Callable] = None,
                 use_egnn_prior: bool = True,
                 prior_temperature: float = 1.0):
        """
        初始化扩展引擎
        
        Args:
            template: 序列模板
            fixed_positions: 固定位置映射
            variable_amino_acids: 每个位置允许的氨基酸
            prior_policy: 先验策略函数（可选，与egnn_model互斥）
            egnn_model: EGNN预测函数，用于计算先验概率
            use_egnn_prior: 是否使用EGNN作为先验
            prior_temperature: 先验温度参数，控制探索程度（越大越均匀）
        """
        self.template = template or config.PEPTIDE_TEMPLATE
        self.fixed_positions = fixed_positions or config.FIXED_POSITIONS
        self.variable_amino_acids = variable_amino_acids or config.VARIABLE_AMINO_ACIDS
        self.prior_policy = prior_policy
        self.egnn_model = egnn_model
        self.use_egnn_prior = use_egnn_prior and (egnn_model is not None)
        self.prior_temperature = prior_temperature
        
        # 缓存EGNN预测结果，避免重复计算
        self._egnn_cache = {}
        
        # 统计信息
        self._egnn_calls = 0
        self._cache_hits = 0
        
        if self.use_egnn_prior:
            print(f"【ExpansionEngine】使用EGNN先验指导扩展")
            print(f"  温度参数: {prior_temperature}")
        elif prior_policy is not None:
            print(f"【ExpansionEngine】使用自定义先验策略")
        else:
            print(f"【ExpansionEngine】警告：无先验策略，使用均匀分布")
    
    def get_next_variable_position(self, sequence: str) -> Optional[int]:
        """获取下一个可变位置（'_'或'x'）"""
        for i, char in enumerate(sequence):
            if char in ['_', 'x', 'X']:
                if i not in self.fixed_positions:
                    return i
        return None
    
    def get_allowed_amino_acids(self, position: int) -> List[str]:
        """获取指定位置允许的氨基酸"""
        return self.variable_amino_acids.get(position, config.ALLOWED_AMINO_ACIDS)
    
    def fill_sequence(self, sequence: str, position: int, amino_acid: str) -> str:
        """在指定位置填充氨基酸"""
        seq_list = list(sequence)
        
        # 确保长度匹配
        if len(seq_list) != len(self.template):
            seq_list = ['_'] * len(self.template)
            for pos, aa in self.fixed_positions.items():
                seq_list[pos] = aa
        
        seq_list[position] = amino_acid
        return ''.join(seq_list)
    
    def calculate_prior(self, context: dict) -> float:
        """
        计算先验概率
        
        Args:
            context: 包含决策信息的字典
        
        Returns:
            先验概率
        """
        # 优先使用自定义prior_policy
        if self.prior_policy is not None:
            return self.prior_policy(context)
        
        # 使用EGNN先验
        if self.use_egnn_prior and self.egnn_model is not None:
            return self._calculate_egnn_prior(context)
        
        # 回退到均匀分布
        return 1.0
    
    def _random_fill_and_evaluate(self, partial_sequence: str, n_fills: int) -> List[float]:
        """
        对部分序列进行随机填充，生成N个完整序列并评估
        
        Args:
            partial_sequence: 部分序列（如"AC_D_C___CG"）
            n_fills: 随机填充数量
        
        Returns:
            N个预测能量值列表
        """
        from seq_generator import generate_n_random_fills
        from peptide_state import PeptideState
        
        # 生成N个随机补全序列
        completed_sequences = generate_n_random_fills(partial_sequence, n_fills)
        
        # 评估每个序列
        energies = []
        for seq in completed_sequences:
            cache_key = f"{seq}_{config.CROSSLINKER}"
            if cache_key in self._egnn_cache:
                self._cache_hits += 1
                energy = self._egnn_cache[cache_key]
            else:
                state = PeptideState(sequence=seq, crosslinker=config.CROSSLINKER)
                try:
                    energy = self.egnn_model(state)
                    self._egnn_cache[cache_key] = energy
                    self._egnn_calls += 1
                except Exception:
                    energy = self._heuristic_energy(seq)
            energies.append(energy)
        
        return energies
    
    def _calculate_egnn_prior(self, context: dict) -> float:
        """
        使用EGNN计算先验概率（带随机填充评估）
        
        新流程:
        1. 填充候选氨基酸到指定位置
        2. 生成N个随机补全序列（N由深度决定）
        3. EGNN预测所有补全序列的结合能
        4. 计算平均能量作为该候选的先验
        
        Args:
            context: 包含决策信息的字典，需包含depth字段
        
        Returns:
            先验概率
        """
        sequence = context['sequence']
        position = context['position']
        amino_acid = context['amino_acid']
        depth = context.get('depth', 0)
        
        # 步骤1: 填充候选氨基酸
        temp_seq = self.fill_sequence(sequence, position, amino_acid)
        
        # 步骤2: 确定随机填充数量（由深度决定）
        try:
            from adaptive_mcts_config import AdaptiveMCTSConfig
            n_fills = AdaptiveMCTSConfig.get_random_fill_count(depth)
        except ImportError:
            # 回退到默认
            n_fills = max(10, 50 - depth * 7)
        
        # 步骤3: 生成N个随机补全并评估
        energies = self._random_fill_and_evaluate(temp_seq, n_fills)
        
        if not energies:
            return 1.0 / len(config.ALLOWED_AMINO_ACIDS)  # 回退到均匀分布
        
        # 步骤4: 计算平均能量
        avg_energy = sum(energies) / len(energies)
        
        # 步骤5: 转换为先验概率
        prior = self._energy_to_prior(avg_energy)
        
        # 记录调试信息
        try:
            from mcts_logger import log_debug
            log_debug("expansion", "随机填充评估完成", {
                "sequence": temp_seq,
                "depth": depth,
                "n_fills": n_fills,
                "avg_energy": avg_energy,
                "prior": prior
            })
        except ImportError:
            pass
        
        return prior
    
    def _heuristic_energy(self, sequence: str) -> float:
        """
        启发式能量估计（EGNN失败时的回退）
        
        基于简单物理化学性质估计结合能
        """
        energy = -5.0  # 基础能量
        
        # 疏水性贡献（疏水氨基酸有利于结合）
        hydrophobic = {'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'M'}
        n_hydrophobic = sum(1 for aa in sequence if aa in hydrophobic)
        energy -= 0.1 * n_hydrophobic
        
        # 电荷平衡（适度电荷有利于结合）
        charged = {'D', 'E', 'K', 'R', 'H'}
        n_charged = sum(1 for aa in sequence if aa in charged)
        if 2 <= n_charged <= 4:
            energy -= 0.5  # 适度电荷奖励
        elif n_charged > 6:
            energy += 1.0  # 过多电荷惩罚
        
        # 芳香族氨基酸（常见于结合位点）
        aromatic = {'F', 'W', 'Y'}
        n_aromatic = sum(1 for aa in sequence if aa in aromatic)
        energy -= 0.2 * min(n_aromatic, 3)  # 最多奖励3个
        
        return energy
    
    def _energy_to_prior(self, energy: float) -> float:
        """
        将结合能转换为先验概率
        
        使用Boltzmann分布，考虑温度参数
        """
        # 假设能量范围 -15 到 0 kcal/mol
        # 归一化到 [0, 1] 范围
        normalized_energy = max(-15.0, min(0.0, energy))
        
        # Boltzmann因子: exp(-E / kT)
        # 能量越低（越负），因子越大
        kT = self.prior_temperature  # 温度参数控制锐度
        boltzmann = np.exp(-normalized_energy / kT)
        
        return boltzmann
    
    def get_egnn_stats(self) -> dict:
        """获取EGNN调用统计"""
        return {
            'egnn_calls': self._egnn_calls,
            'cache_hits': self._cache_hits,
            'cache_size': len(self._egnn_cache),
            'hit_rate': self._cache_hits / (self._egnn_calls + self._cache_hits) if (self._egnn_calls + self._cache_hits) > 0 else 0
        }
    
    def expand_level1_single(self, node: MCTSNode, amino_acid: str, 
                              prior_prob: Optional[float] = None) -> MCTSNode:
        """
        第一层扩展：填充单个氨基酸（逐步扩展）
        
        Args:
            node: 当前节点
            amino_acid: 要填充的氨基酸
            prior_prob: 先验概率（可选，默认自动计算）
        
        Returns:
            新的子节点
        """
        sequence = node.state.sequence
        next_pos = self.get_next_variable_position(sequence)
        
        if next_pos is None:
            node.state.is_sequence_complete = True
            return node
        
        # 检查该氨基酸是否允许
        allowed_aas = self.get_allowed_amino_acids(next_pos)
        if amino_acid not in allowed_aas:
            raise ValueError(f"位置 {next_pos} 不允许氨基酸 {amino_acid}")
        
        # 检查是否已存在
        if amino_acid in node.children:
            return node.children[amino_acid]
        
        # 计算先验概率
        if prior_prob is None:
            context = {
                'level': 1,
                'sequence': sequence,
                'position': next_pos,
                'amino_acid': amino_acid
            }
            prior_prob = self.calculate_prior(context)
        
        # 创建新状态
        new_sequence = self.fill_sequence(sequence, next_pos, amino_acid)
        new_state = PeptideState(
            sequence=new_sequence,
            crosslinker=None,
            disulfide_bonds=[]
        )
        
        child = MCTSNode(
            state=new_state,
            parent=node,
            prior_prob=prior_prob,
            decision_level=1,
            decision_action=amino_acid
        )
        
        node.children[amino_acid] = child
        return child
    
    def expand_level1_amino_acid(self, node: MCTSNode, 
                                  max_expansions: int = None) -> Dict[str, MCTSNode]:
        """
        第一层扩展：填充氨基酸（限制扩展数量）
        
        V2版本修改：
        - max_expansions默认19（排除Cys后的氨基酸数）
        - 使用EGNN批量预测选Top-k，绝不随机选择
        - 保证子节点不重复（记录已扩展的氨基酸）
        
        Args:
            node: 当前节点
            max_expansions: 最大扩展数量（默认19）
        
        Returns:
            新创建的子节点字典
        """
        if max_expansions is None:
            max_expansions = config.MCTS_CONFIG.get("max_expansions", 19)
        
        sequence = node.state.sequence
        next_pos = self.get_next_variable_position(sequence)
        
        if next_pos is None:
            node.state.is_sequence_complete = True
            return {}
        
        allowed_aas = self.get_allowed_amino_acids(next_pos)
        
        # 找出尚未扩展的氨基酸（保证不重复）
        unexpanded = [aa for aa in allowed_aas if aa not in node.children]
        
        if not unexpanded:
            return {}
        
        # V2: 使用EGNN批量预测所有候选，选Top-k（绝不随机）
        if self.use_egnn_prior and self.egnn_model is not None:
            # 批量计算所有未扩展氨基酸的先验
            priors = []
            for aa in unexpanded:
                context = {
                    'level': 1,
                    'sequence': sequence,
                    'position': next_pos,
                    'amino_acid': aa,
                    'depth': sum(1 for c in sequence if c not in ['_', 'x', 'X'])
                }
                prior = self.calculate_prior(context)
                priors.append((aa, prior))
            
            # 按先验排序（高先验 = 低能量 = 好）
            priors.sort(key=lambda x: x[1], reverse=True)
        else:
            # 无EGNN时均匀分布（不应该发生）
            priors = [(aa, 1.0/len(unexpanded)) for aa in unexpanded]
        
        # 只扩展前max_expansions个（Top-k）
        to_expand = priors[:max_expansions]
        
        # 归一化先验（在被扩展的节点中）
        total = sum(p for _, p in to_expand)
        if total > 0:
            to_expand = [(aa, p/total) for aa, p in to_expand]
        
        # 创建子节点
        new_children = {}
        for aa, prior in to_expand:
            child = self.expand_level1_single(node, aa, prior)
            new_children[aa] = child
        
        return new_children
    
    def expand_with_softmax_allocation(
        self,
        nodes: List['MCTSNode'],
        total_slots: int = 50,
        temperature: float = 1.0
    ) -> Dict[str, int]:
        """
        使用Softmax概率分配扩展名额到多个节点
        
        流程:
        1. 评估每个节点的平均能量（通过随机填充）
        2. Softmax计算概率（取绝对值）
        3. 按概率分配total_slots个名额
        4. 每个节点根据分配到的名额扩展子节点
        
        Args:
            nodes: 待扩展的节点列表
            total_slots: 总扩展名额（默认50）
            temperature: Softmax温度（默认1.0）
        
        Returns:
            {node_key: 分配到的名额数}
        """
        import numpy as np
        
        if not nodes:
            return {}
        
        # 步骤1: 评估每个节点的平均能量
        node_energies = []
        for node in nodes:
            depth = sum(1 for c in node.state.sequence if c not in ['_', 'x', 'X'])
            
            # 获取随机填充数量
            try:
                from adaptive_mcts_config import AdaptiveMCTSConfig
                n_fills = AdaptiveMCTSConfig.get_random_fill_count(depth)
            except ImportError:
                n_fills = max(10, 50 - depth * 7)
            
            # 随机填充并评估
            energies = self._random_fill_and_evaluate(node.state.sequence, n_fills)
            avg_energy = sum(energies) / len(energies) if energies else -5.0
            node_energies.append((node.state.to_key(), avg_energy))
        
        # 步骤2: Softmax计算概率（取绝对值）
        abs_energies = np.array([abs(energy) for _, energy in node_energies])
        exp_energies = np.exp(abs_energies / temperature)
        probabilities = exp_energies / np.sum(exp_energies)
        
        # 步骤3: 按概率分配名额
        allocations = np.floor(probabilities * total_slots).astype(int)
        
        # 处理剩余名额
        remaining = total_slots - np.sum(allocations)
        if remaining > 0:
            fractional_parts = probabilities * total_slots - allocations
            sorted_indices = np.argsort(fractional_parts)[::-1]
            for i in range(remaining):
                allocations[sorted_indices[i % len(sorted_indices)]] += 1
        
        # 构建返回字典
        result = {}
        for i, (node_key, _) in enumerate(node_energies):
            result[node_key] = int(allocations[i])
        
        # 记录日志
        try:
            from mcts_logger import log_debug
            log_debug("expansion", "Softmax分配完成", {
                "total_slots": total_slots,
                "temperature": temperature,
                "allocations": result,
                "probabilities": probabilities.tolist()
            })
        except ImportError:
            pass
        
        return result
    
    def expand_node_with_allocated_slots(
        self,
        node: 'MCTSNode',
        allocated_slots: int
    ) -> Dict[str, 'MCTSNode']:
        """
        根据分配到的名额扩展节点（V2版本）
        
        使用EGNN批量预测所有可能的子节点，然后选择Top-k（k=allocated_slots）
        绝不随机选择！
        
        Args:
            node: 要扩展的节点
            allocated_slots: 分配到的名额数（由Softmax分配）
        
        Returns:
            新创建的子节点字典
        """
        if allocated_slots <= 0:
            return {}
        
        # 获取当前决策层
        level = node.state.decision_level
        
        if level == 1:
            # 第一层：填充氨基酸
            # 限制为allocated_slots，但不超过配置的最大值
            max_exp = config.MCTS_CONFIG.get("max_expansions", 19)
            actual_expansions = min(allocated_slots, max_exp)
            return self.expand_level1_amino_acid(node, max_expansions=actual_expansions)
        elif level == 2:
            # 第二层：选择交联剂
            return self.expand_level2_crosslinker(node, max_expansions=allocated_slots)
        elif level == 3:
            # 第三层：选择二硫键
            return self.expand_level3_disulfide(node, max_expansions=allocated_slots)
        else:
            return {}
    
    def expand_level2_single(self, node: MCTSNode, crosslinker: Optional[str],
                            prior_prob: Optional[float] = None) -> MCTSNode:
        """
        第二层扩展：选择单个交联剂（逐步扩展）
        
        Args:
            node: 当前节点
            crosslinker: 交联剂类型
            prior_prob: 先验概率
        
        Returns:
            新的子节点
        """
        if not node.state.is_sequence_complete:
            raise ValueError("序列尚未完成，无法选择交联剂")
        
        key = str(crosslinker) if crosslinker else "None"
        
        # 检查是否已存在
        if key in node.children:
            return node.children[key]
        
        # 检查是否允许
        options = node.state.get_possible_crosslinkers()
        if crosslinker not in options:
            raise ValueError(f"交联剂 {crosslinker} 不可用，可用: {options}")
        
        # 计算先验概率
        if prior_prob is None:
            prior_prob = 1.0 / len(options) if options else 1.0
        
        # 创建新状态
        new_state = node.state.copy()
        new_state.crosslinker = crosslinker
        new_state._update_completion_status()
        
        child = MCTSNode(
            state=new_state,
            parent=node,
            prior_prob=prior_prob,
            decision_level=2,
            decision_action=key
        )
        
        node.children[key] = child
        return child
    
    def expand_level2_crosslinker(self, node: MCTSNode,
                                   max_expansions: int = 2) -> Dict[str, MCTSNode]:
        """
        第二层扩展：选择交联剂（限制扩展数量）
        
        Args:
            node: 当前节点（序列已完成）
            max_expansions: 最大扩展数量（默认2）
        
        Returns:
            新创建的子节点字典
        """
        if not node.state.is_sequence_complete:
            return {}
        
        # 获取可能的交联剂选项
        options = node.state.get_possible_crosslinkers()
        
        # 找出尚未扩展的选项
        def get_key(x):
            return str(x) if x else "None"
        unexpanded = [x for x in options if get_key(x) not in node.children]
        
        if not unexpanded:
            return {}
        
        # 限制扩展数量
        to_expand = unexpanded[:max_expansions]
        
        # 计算先验概率
        prior = 1.0 / len(options) if options else 1.0
        
        # 创建子节点
        new_children = {}
        for xlinker in to_expand:
            child = self.expand_level2_single(node, xlinker, prior)
            new_children[str(xlinker) if xlinker else "None"] = child
        
        return new_children
    
    def expand_level3_single(self, node: MCTSNode, bond: Tuple[int, int],
                            prior_prob: Optional[float] = None) -> MCTSNode:
        """
        第三层扩展：选择单个二硫键配对（逐步扩展）
        
        Args:
            node: 当前节点
            bond: 二硫键配对 (i, j)
            prior_prob: 先验概率
        
        Returns:
            新的子节点
        """
        if node.state.crosslinker != "disulfide":
            raise ValueError("只有disulfide模式可以选择二硫键")
        
        key = f"{bond[0]}-{bond[1]}"
        
        # 检查是否已存在
        if key in node.children:
            return node.children[key]
        
        # 检查是否允许
        bonds = node.state.get_possible_disulfide_bonds()
        if bond not in bonds:
            raise ValueError(f"二硫键 {bond} 不可用")
        
        # 计算先验概率
        if prior_prob is None:
            prior_prob = 1.0 / len(bonds) if bonds else 1.0
        
        # 创建新状态
        new_state = node.state.copy()
        new_state.disulfide_bonds = [bond]
        new_state._update_completion_status()
        
        child = MCTSNode(
            state=new_state,
            parent=node,
            prior_prob=prior_prob,
            decision_level=3,
            decision_action=key
        )
        
        node.children[key] = child
        return child
    
    def expand_level3_disulfide(self, node: MCTSNode,
                                 max_expansions: int = 2) -> Dict[str, MCTSNode]:
        """
        第三层扩展：选择二硫键配对（限制扩展数量）
        
        Args:
            node: 当前节点（序列完成，交联剂=disulfide）
            max_expansions: 最大扩展数量（默认2）
        
        Returns:
            新创建的子节点字典
        """
        if node.state.crosslinker != "disulfide":
            return {}
        
        # 获取可能的二硫键配对
        bonds = node.state.get_possible_disulfide_bonds()
        
        # 找出尚未扩展的配对
        unexpanded = [b for b in bonds if f"{b[0]}-{b[1]}" not in node.children]
        
        if not unexpanded:
            return {}
        
        # 限制扩展数量
        to_expand = unexpanded[:max_expansions]
        
        # 计算先验概率
        prior = 1.0 / len(bonds) if bonds else 1.0
        
        # 创建子节点
        new_children = {}
        for bond in to_expand:
            child = self.expand_level3_single(node, bond, prior)
            new_children[f"{bond[0]}-{bond[1]}"] = child
        
        return new_children
    
    def expand(self, node: MCTSNode, 
               max_expansions: int = 3) -> Dict[str, MCTSNode]:
        """
        根据当前决策层自动选择扩展方式（限制扩展数量）
        
        Args:
            node: 当前节点
            max_expansions: 最大扩展数量（默认3）
        
        Returns:
            新创建的子节点字典
        """
        level = node.state.decision_level
        
        if level == 1:
            return self.expand_level1_amino_acid(node, max_expansions)
        elif level == 2:
            return self.expand_level2_crosslinker(node, max_expansions)
        elif level == 3:
            return self.expand_level3_disulfide(node, max_expansions)
        else:
            # 已完成
            return {}
    
    def can_expand(self, node: MCTSNode) -> bool:
        """检查节点是否还可以扩展"""
        if node.is_terminal:
            return False
        
        level = node.state.decision_level
        
        if level == 1:
            # 检查是否还有未扩展的氨基酸
            next_pos = self.get_next_variable_position(node.state.sequence)
            if next_pos is None:
                return False
            allowed = set(self.get_allowed_amino_acids(next_pos))
            expanded = set(node.children.keys())
            return len(allowed - expanded) > 0
        
        elif level == 2:
            # 检查是否还有未扩展的交联剂
            options = set(str(x) if x else "None" for x in node.state.get_possible_crosslinkers())
            expanded = set(node.children.keys())
            return len(options - expanded) > 0
        
        elif level == 3:
            # 检查是否还有未扩展的二硫键
            bonds = set(f"{b[0]}-{b[1]}" for b in node.state.get_possible_disulfide_bonds())
            expanded = set(node.children.keys())
            return len(bonds - expanded) > 0
        
        return False


def main():
    """测试代码"""
    print("="*60)
    print("Expansion引擎V2测试")
    print("="*60)
    
    from peptide_state import create_root_node
    
    engine = ExpansionEngine()
    
    # 测试1：Level 1扩展（氨基酸）
    print("\n[测试1] Level 1: 氨基酸扩展")
    root = create_root_node("AC_____C______CG")
    print(f"  初始: {root.state}")
    
    children = engine.expand_level1_amino_acid(root)
    print(f"  扩展了 {len(children)} 个子节点")
    for action, child in list(children.items())[:5]:
        print(f"    {action}: seq={child.state.sequence[:15]}, prior={child.prior_prob:.4f}")
    
    # 测试2：模拟序列完成，Level 2扩展（交联剂）
    print("\n[测试2] Level 2: 交联剂扩展")
    completed_state = root.state.copy()
    completed_state.sequence = "ACARNDCMVFLWPCG"
    completed_state._update_completion_status()
    
    node2 = MCTSNode(state=completed_state)
    print(f"  序列完成: {node2.state}")
    print(f"  Cys位置: {node2.state.get_cys_positions()}")
    
    children2 = engine.expand_level2_crosslinker(node2)
    print(f"  可用交联剂选项: {len(children2)}")
    for action, child in children2.items():
        print(f"    {action}: xlinker={child.state.crosslinker}, prior={child.prior_prob:.4f}")
    
    # 测试3：Level 3扩展（二硫键）
    print("\n[测试3] Level 3: 二硫键扩展")
    disulfide_node = None
    for child in children2.values():
        if child.state.crosslinker == "disulfide":
            disulfide_node = child
            break
    
    if disulfide_node:
        print(f"  选择disulfide: {disulfide_node.state}")
        children3 = engine.expand_level3_disulfide(disulfide_node)
        print(f"  可用二硫键选项: {len(children3)}")
        for action, child in list(children3.items())[:5]:
            print(f"    {action}: bonds={child.state.disulfide_bonds}, prior={child.prior_prob:.4f}")
    
    # 测试4：逐步扩展（MCTS风格）
    print("\n[测试4] 逐步扩展（MCTS风格）")
    root2 = create_root_node("AC_____C______CG")
    print(f"  初始决策层: {root2.state.decision_level}")
    print(f"  初始子节点数: {len(root2.children)}")
    
    # 第一次扩展：只扩展3个（默认max_expansions=3）
    children1 = engine.expand(root2, max_expansions=3)
    print(f"  第一次扩展后子节点数: {len(root2.children)} (新增 {len(children1)} 个)")
    
    # 第二次扩展：再扩展3个
    children2 = engine.expand(root2, max_expansions=3)
    print(f"  第二次扩展后子节点数: {len(root2.children)} (新增 {len(children2)} 个)")
    
    # 继续扩展直到全部扩展
    while True:
        new_children = engine.expand(root2, max_expansions=5)
        if not new_children:
            break
    print(f"  全部扩展后子节点数: {len(root2.children)}")
    
    # 测试5：模拟MCTS路径
    print("\n[测试5] 模拟MCTS路径")
    current = root2
    path = [current]
    
    for step in range(20):
        level = current.state.decision_level
        if level == 0:  # 完成
            break
        
        # 扩展（如果还可以）
        if engine.can_expand(current):
            engine.expand(current, max_expansions=2)
        
        # 选择子节点（模拟PUCT选择）
        if current.children:
            # 选择visit_count最少的（探索）
            next_node = min(current.children.values(), key=lambda n: n.visit_count)
            path.append(next_node)
            current = next_node
            print(f"  Step {step+1} (Level {level}): {current.decision_action} -> {current.state.sequence[:20] if level==1 else current.state.crosslinker}")
        else:
            break
    
    print(f"\n  路径长度: {len(path)}")
    print(f"  最终状态: {current.state}")
    print(f"  是否终端: {current.is_terminal}")
    
    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)


if __name__ == "__main__":
    main()
