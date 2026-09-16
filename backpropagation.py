#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Backpropagation模块 V2 (backpropagation.py)
支持三层决策的回溯引擎
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from peptide_state import MCTSNode


class BackpropagationEngine:
    """
    MCTS回溯引擎 V2
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def backpropagate(self, path: List[MCTSNode], reward: float) -> None:
        """
        回溯更新路径上所有节点的统计信息
        
        Args:
            path: 从根节点到叶节点的完整路径
            reward: Simulation返回的归一化分数 [0, 1]
        """
        if not path:
            raise ValueError("路径不能为空")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Backpropagation: 更新 {len(path)} 个节点")
            print(f"Reward: {reward:.4f}")
            print(f"{'='*60}")
        
        for i, node in enumerate(path):
            old_visits = node.visit_count
            old_total = node.total_score
            
            node.visit_count += 1
            node.total_score += reward
            
            if self.verbose:
                print(f"  [{i}] Level {node.decision_level}: {node.decision_action}")
                print(f"      visits: {old_visits} -> {node.visit_count}")
                print(f"      avg_score: {node.average_score:.4f}")
        
        if self.verbose:
            print(f"{'='*60}\n")
    
    def get_node_stats(self, node: MCTSNode) -> Dict[str, Any]:
        """获取节点统计信息"""
        return {
            'sequence': node.state.sequence,
            'crosslinker': node.state.crosslinker,
            'disulfide_bonds': node.state.disulfide_bonds,
            'visit_count': node.visit_count,
            'total_score': node.total_score,
            'average_score': node.average_score,
            'is_terminal': node.is_terminal,
            'num_children': len(node.children),
            'decision_level': node.decision_level,
            'decision_action': node.decision_action
        }


def main():
    """测试代码"""
    print("="*60)
    print("Backpropagation引擎V2测试")
    print("="*60)
    
    from peptide_state import MCTSNode, PeptideState
    
    # 创建引擎
    engine = BackpropagationEngine(verbose=True)
    
    # 创建测试路径
    # 根节点 -> 氨基酸A -> 交联剂TBMB
    root_state = PeptideState(sequence="AC_____C______CG")
    root = MCTSNode(state=root_state, decision_level=0)
    
    state1 = PeptideState(sequence="ACA____C______CG")
    node1 = MCTSNode(state=state1, parent=root, decision_level=1, decision_action="A")
    root.children["A"] = node1
    
    state2 = PeptideState(sequence="ACARNDCMVFLWPCG")
    node2 = MCTSNode(state=state2, parent=node1, decision_level=1, decision_action="R")
    node1.children["R"] = node2
    
    state3 = PeptideState(sequence="ACARNDCMVFLWPCG", crosslinker="TBMB")
    node3 = MCTSNode(state=state3, parent=node2, decision_level=2, decision_action="TBMB")
    node2.children["TBMB"] = node3
    
    path = [root, node1, node2, node3]
    
    # 执行回溯
    reward = 0.85
    engine.backpropagate(path, reward)
    
    # 验证
    print("\n验证结果:")
    for i, node in enumerate(path):
        print(f"  节点 {i}: visits={node.visit_count}, avg={node.average_score:.4f}")
    
    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)


if __name__ == "__main__":
    main()
