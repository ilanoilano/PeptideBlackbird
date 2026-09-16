#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肽状态定义 (peptide_state.py)

扩展MCTS以支持三层决策：
1. 氨基酸序列填充
2. 交联剂/环化方式选择
3. 二硫键配对（如果使用二硫键）

状态表示：
- sequence: 氨基酸序列（可能含占位符'_'）
- crosslinker: 交联剂类型（TBMB/TATA/TBAB/None/disulfide）
- disulfide_bonds: 二硫键配对列表
- decision_level: 当前决策层（1=序列, 2=交联剂, 3=二硫键）
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

import config


# 支持的交联剂类型
CROSSLINKER_TYPES = ["TBMB", "TATA", "TBAB", "disulfide", None]

# 交联剂需要的Cys数量
CROSSLINKER_CYS_COUNT = {
    "TBMB": 3,
    "TATA": 3,
    "TBAB": 4,
    "disulfide": 2,  # 每对二硫键需要2个Cys
    None: 0,
}


@dataclass
class PeptideState:
    """
    肽状态 - 包含序列和拓扑信息
    
    用于MCTS节点，支持三层决策
    """
    sequence: str = ""                           # 氨基酸序列（可能含'_'）
    crosslinker: Optional[str] = None            # 交联剂类型
    disulfide_bonds: List[Tuple[int, int]] = field(default_factory=list)  # 二硫键配对
    
    # 决策层标记
    is_sequence_complete: bool = False           # 序列是否填完
    is_topology_complete: bool = False           # 拓扑是否确定
    
    def __post_init__(self):
        """初始化后检查状态一致性"""
        if not self.sequence:
            self.sequence = config.PEPTIDE_TEMPLATE.replace('x', '_')
        
        # 自动判断完成状态
        self._update_completion_status()
    
    def _update_completion_status(self):
        """更新完成状态"""
        # 序列完成：没有占位符
        self.is_sequence_complete = '_' not in self.sequence and 'x' not in self.sequence.lower()
        
        # 拓扑完成：
        # - 如果选择了交联剂（TBMB/TATA/TBAB），拓扑自动完成
        # - 如果选择了disulfide，需要至少一对二硫键
        # - 如果选择了None（线性肽），拓扑自动完成
        if self.crosslinker in ["TBMB", "TATA", "TBAB", None]:
            self.is_topology_complete = True
        elif self.crosslinker == "disulfide":
            self.is_topology_complete = len(self.disulfide_bonds) > 0
        else:
            self.is_topology_complete = False
    
    @property
    def is_terminal(self) -> bool:
        """是否是终端状态（序列和拓扑都完成）"""
        return self.is_sequence_complete and self.is_topology_complete
    
    @property
    def decision_level(self) -> int:
        """
        当前决策层
        
        Returns:
            1: 填充氨基酸序列
            2: 选择交联剂/环化方式
            3: 选择二硫键配对（仅当crosslinker="disulfide"）
        """
        if not self.is_sequence_complete:
            return 1  # 填充序列
        elif not self.is_topology_complete:
            if self.crosslinker is None:
                return 2  # 需要选择交联剂
            elif self.crosslinker == "disulfide" and not self.disulfide_bonds:
                return 3  # 需要选择二硫键
        return 0  # 已完成
    
    def get_cys_positions(self) -> List[int]:
        """获取序列中所有Cys的位置"""
        return [i for i, aa in enumerate(self.sequence) if aa == 'C']
    
    def get_possible_crosslinkers(self) -> List[Optional[str]]:
        """
        获取可能的交联剂选项
        
        根据Cys数量决定可用选项
        """
        cys_count = len(self.get_cys_positions())
        options = []
        
        for xlinker, required_cys in CROSSLINKER_CYS_COUNT.items():
            if cys_count >= required_cys:
                options.append(xlinker)
        
        return options
    
    def get_possible_disulfide_bonds(self) -> List[Tuple[int, int]]:
        """
        获取可能的二硫键配对
        
        Returns:
            所有可能的Cys配对组合
        """
        cys_positions = self.get_cys_positions()
        bonds = []
        
        # 生成所有可能的配对
        for i in range(len(cys_positions)):
            for j in range(i + 1, len(cys_positions)):
                bonds.append((cys_positions[i], cys_positions[j]))
        
        return bonds
    
    def copy(self) -> 'PeptideState':
        """创建副本"""
        return PeptideState(
            sequence=self.sequence,
            crosslinker=self.crosslinker,
            disulfide_bonds=self.disulfide_bonds.copy(),
            is_sequence_complete=self.is_sequence_complete,
            is_topology_complete=self.is_topology_complete
        )
    
    def to_key(self) -> str:
        """转换为唯一标识符（用于字典key）"""
        bonds_str = "|".join(f"{i}-{j}" for i, j in self.disulfide_bonds)
        return f"{self.sequence}_{self.crosslinker}_{bonds_str}"
    
    def __repr__(self) -> str:
        level = self.decision_level
        level_str = {0: "Complete", 1: "Seq", 2: "Xlinker", 3: "Disulfide"}.get(level, "Unknown")
        return (f"PeptideState(seq='{self.sequence}', xlinker={self.crosslinker}, "
                f"bonds={self.disulfide_bonds}, level={level_str})")


@dataclass
class MCTSNode:
    """
    MCTS树节点 - 使用PeptideState
    
    支持三层决策的完整状态
    """
    state: PeptideState                           # 肽状态（序列+拓扑）
    visit_count: int = 0
    total_score: float = 0.0
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    parent: Optional['MCTSNode'] = None
    prior_prob: float = 1.0                       # 先验概率 P(s,a)
    
    # 决策信息
    decision_action: Optional[str] = None         # 导致此节点的动作（氨基酸/交联剂/二硫键）
    decision_level: int = 1                       # 此节点对应的决策层
    
    @property
    def average_score(self) -> float:
        """计算平均得分（Q值）"""
        if self.visit_count == 0:
            return 0.0
        return self.total_score / self.visit_count
    
    @property
    def is_terminal(self) -> bool:
        """是否是终端节点"""
        return self.state.is_terminal
    
    def __repr__(self) -> str:
        return (f"MCTSNode(state={self.state}, visits={self.visit_count}, "
                f"avg={self.average_score:.4f}, prior={self.prior_prob:.4f})")


def create_root_node(template: Optional[str] = None) -> MCTSNode:
    """
    创建MCTS根节点
    
    Args:
        template: 序列模板（默认从config读取）
    
    Returns:
        根节点
    """
    if template is None:
        template = config.PEPTIDE_TEMPLATE
    
    # 将模板中的'x'替换为'_'
    sequence = template.replace('x', '_').replace('X', '_')
    
    state = PeptideState(sequence=sequence)
    
    return MCTSNode(
        state=state,
        visit_count=0,
        total_score=0.0,
        children={},
        parent=None,
        prior_prob=1.0,
        decision_level=1
    )


def main():
    """测试代码"""
    print("="*60)
    print("PeptideState测试")
    print("="*60)
    
    # 测试1：初始状态
    print("\n[测试1] 初始状态")
    state = PeptideState()
    print(f"  初始: {state}")
    print(f"  决策层: {state.decision_level}")
    print(f"  是否终端: {state.is_terminal}")
    
    # 测试2：序列完成后的状态
    print("\n[测试2] 序列完成后")
    state.sequence = "ACARNDCMVFLWPCG"
    state._update_completion_status()
    print(f"  序列完成: {state}")
    print(f"  决策层: {state.decision_level}")
    print(f"  Cys位置: {state.get_cys_positions()}")
    print(f"  可用交联剂: {state.get_possible_crosslinkers()}")
    
    # 测试3：选择交联剂后
    print("\n[测试3] 选择TBMB后")
    state.crosslinker = "TBMB"
    state._update_completion_status()
    print(f"  选择TBMB: {state}")
    print(f"  决策层: {state.decision_level}")
    print(f"  是否终端: {state.is_terminal}")
    
    # 测试4：选择二硫键的情况
    print("\n[测试4] 选择disulfide")
    state2 = PeptideState(sequence="ACARNDCMVFLWPCG")
    state2.crosslinker = "disulfide"
    state2._update_completion_status()
    print(f"  选择disulfide: {state2}")
    print(f"  决策层: {state2.decision_level}")
    print(f"  可能的二硫键: {state2.get_possible_disulfide_bonds()[:5]}...")
    
    # 添加二硫键
    bonds = state2.get_possible_disulfide_bonds()
    if bonds:
        state2.disulfide_bonds = [bonds[0]]
        state2._update_completion_status()
        print(f"  添加二硫键后: {state2}")
        print(f"  决策层: {state2.decision_level}")
        print(f"  是否终端: {state2.is_terminal}")
    
    # 测试5：MCTSNode
    print("\n[测试5] MCTSNode")
    root = create_root_node()
    print(f"  根节点: {root}")
    print(f"  状态: {root.state}")
    
    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)


if __name__ == "__main__":
    main()
