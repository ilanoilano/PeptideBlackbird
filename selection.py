#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Selection模块 V2 (selection.py)
支持三层决策的PUCT选择器

决策层：
1. 氨基酸序列填充
2. 交联剂/环化方式选择
3. 二硫键配对选择

PUCT公式：
PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
"""

import math
import random
from typing import Optional, Dict, Callable

from peptide_state import MCTSNode, PeptideState


class PUCTSelector:
    """
    PUCT选择器 - 支持三层决策
    """
    
    def __init__(self, c_puct: float = 1.5, epsilon: float = 0.0):
        """
        初始化PUCT选择器
        
        Args:
            c_puct: 探索常数（默认1.5）
            epsilon: epsilon-贪婪探索概率
        """
        self.c_puct = c_puct
        self.epsilon = epsilon
    
    def calculate_puct(self, child: MCTSNode, parent_visits: int) -> float:
        """
        计算PUCT分数
        
        PUCT(s,a) = Q(s,a) + U(s,a)
        U(s,a) = c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
        """
        # Q值：平均得分
        q_value = child.average_score
        
        # U值：探索奖励
        if parent_visits == 0:
            u_value = float('inf')
        else:
            u_value = (self.c_puct * child.prior_prob * 
                      math.sqrt(parent_visits) / (1 + child.visit_count))
        
        return q_value + u_value
    
    def select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        使用PUCT公式选择最佳子节点
        """
        if not node.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        parent_visits = node.visit_count
        
        for action, child in node.children.items():
            puct_score = self.calculate_puct(child, parent_visits)
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child
        
        return best_child
    
    def select_with_random(self, node: MCTSNode) -> Optional[MCTSNode]:
        """带随机探索的PUCT选择"""
        if not node.children:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(list(node.children.values()))
        else:
            return self.select(node)
    
    def select_path(self, root: MCTSNode,
                    can_expand_fn: Optional[Callable] = None) -> list:
        """
        从根节点选择到叶节点的完整路径
        
        Args:
            root: 根节点
            can_expand_fn: 函数 f(node) -> bool，检查节点是否还可以扩展
        
        Returns:
            节点路径列表
        """
        path = [root]
        current = root
        
        while True:
            # 如果当前节点是终端节点，停止
            if current.is_terminal:
                break
            
            # 如果没有子节点，停止（需要扩展）
            if not current.children:
                break
            
            # 检查是否还可以扩展
            if can_expand_fn and can_expand_fn(current):
                break
            
            # 选择子节点
            next_node = self.select(current)
            
            if next_node is None:
                break
            
            path.append(next_node)
            current = next_node
            
            # 防止无限循环
            if len(path) > 1000:
                print("警告: 选择路径过长")
                break
        
        return path


def main():
    """测试代码"""
    print("="*60)
    print("PUCT选择器V2测试")
    print("="*60)
    
    from peptide_state import create_root_node, PeptideState
    
    # 创建根节点
    root = create_root_node("AC_____C______CG")
    print(f"\n根节点: {root}")
    print(f"  决策层: {root.state.decision_level}")
    
    # 创建选择器
    selector = PUCTSelector(c_puct=1.5)
    
    # 模拟扩展一些子节点（不同决策层）
    # Level 1: 氨基酸
    for aa in ['A', 'D', 'E']:
        state = PeptideState(sequence=f"AC{aa}____C______CG")
        child = MCTSNode(
            state=state,
            parent=root,
            prior_prob=1.0/19,  # 均匀分布
            decision_level=1,
            decision_action=aa
        )
        root.children[aa] = child
    
    root.visit_count = 10
    root.children['A'].visit_count = 5
    root.children['A'].total_score = 3.5
    root.children['D'].visit_count = 3
    root.children['D'].total_score = 2.1
    root.children['E'].visit_count = 2
    root.children['E'].total_score = 1.8
    
    print("\n子节点:")
    for action, child in root.children.items():
        puct = selector.calculate_puct(child, root.visit_count)
        print(f"  {action}: visits={child.visit_count}, avg={child.average_score:.4f}, "
              f"prior={child.prior_prob:.4f}, PUCT={puct:.4f}")
    
    # 测试选择
    print("\n选择测试:")
    for i in range(5):
        selected = selector.select(root)
        print(f"  {i+1}. 选中: {selected.decision_action}")
    
    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)


if __name__ == "__main__":
    main()
