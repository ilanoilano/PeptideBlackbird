#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Selection模块 V3 (selection.py)
正统 MCTS 概率采样选择器
"""

import math
import random
from typing import Optional, Dict, Callable, List

import numpy as np

from peptide_state import MCTSNode, PeptideState


class PUCTSelector:
    """正统 MCTS PUCT 选择器"""

    def __init__(self, c_puct: float = 1.5,
                 temperature: float = 1.0,
                 sample: bool = True,
                 epsilon: float = 0.0):
        self.c_puct = c_puct
        self.temperature = temperature
        self.sample = sample
        self.epsilon = epsilon
        self._select_calls = 0

    def calculate_puct(self, child: MCTSNode, parent_visits: int) -> float:
        """PUCT 分数（Q + U），未访问的 child 的 U 会因分母 (1+0)=1 而较大"""
        q_value = child.average_score
        if parent_visits <= 0:
            u_value = 0.0
        else:
            u_value = (self.c_puct * child.prior_prob *
                       math.sqrt(parent_visits) / (1 + child.visit_count))
        return q_value + u_value


    def select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        选择子节点
        
        分三层策略：
        1. 若有"未访问的子节点" → 从其中按先验采样
        2. 若所有子节点都访问过 → PUCT 概率采样（softmax）
        3. 无子节点返回 None
        """
        if not node.children:
            return None
        
        self._select_calls += 1
        children = list(node.children.values())
        
        # ---- 第一层：未访问优先 ----
        unvisited = [c for c in children if c.visit_count == 0]
        if unvisited:
            priors = np.array([max(c.prior_prob, 1e-8) for c in unvisited], dtype=np.float64)
            s = priors.sum()
            if s > 0:
                probs = priors / s
            else:
                probs = np.ones(len(unvisited)) / len(unvisited)
            idx = int(np.random.choice(len(unvisited), p=probs))
            return unvisited[idx]
        
        # ---- 第二层：PUCT 概率采样 ----
        parent_visits = node.visit_count
        scores = np.array(
            [self.calculate_puct(c, parent_visits) for c in children],
            dtype=np.float64
        )
        
        # ε-贪婪
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(children)
        
        temp = max(self.temperature, 1e-6)
        scores = scores / temp
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        denom = exp_scores.sum()
        if denom <= 0 or not np.all(np.isfinite(exp_scores)):
            idx = int(np.argmax(scores))
            return children[idx]
        probs = exp_scores / denom
        idx = int(np.random.choice(len(children), p=probs))
        return children[idx]


    def select_path(self, root: MCTSNode,
                    can_expand_fn: Optional[Callable] = None,
                    max_depth: int = 100) -> List[MCTSNode]:
        path = [root]
        current = root
        visited = {id(root)}

        for _ in range(max_depth):
            if current.is_terminal:
                break
            if not current.children:
                break
            if can_expand_fn and can_expand_fn(current):
                break

            next_node = self.select(current)
            if next_node is None:
                break

            if id(next_node) in visited:
                print(f"[PUCTSelector] 检测到环，停止 select_path")
                break
            visited.add(id(next_node))

            path.append(next_node)
            current = next_node
        else:
            print(f"[PUCTSelector] 达到最大深度 {max_depth}，停止 select_path")

        return path

    def select_with_random(self, node: MCTSNode) -> Optional[MCTSNode]:
        return self.select(node)

    def get_stats(self) -> dict:
        return {
            "select_calls": self._select_calls,
            "c_puct": self.c_puct,
            "temperature": self.temperature,
            "sample": self.sample,
        }


def main():
    print("=" * 60)
    print("PUCT选择器 V3 测试（正统 MCTS）")
    print("=" * 60)

    from peptide_state import create_root_node

    root = create_root_node()
    print(f"\n根节点: {root.state.sequence}")

    selector = PUCTSelector(c_puct=1.5, sample=True, temperature=1.0)

    for aa in ['A', 'D', 'E', 'F', 'G']:
        state = PeptideState(sequence=f"AC{aa}__C___CG")
        child = MCTSNode(
            state=state,
            parent=root,
            prior_prob=1.0 / 5,
            decision_level=1,
            decision_action=aa,
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

    print("\n概率采样测试（20 次）:")
    counts = {}
    for i in range(20):
        selected = selector.select(root)
        counts[selected.decision_action] = counts.get(selected.decision_action, 0) + 1
    for action, cnt in sorted(counts.items()):
        print(f"  {action}: {cnt}/20")

    print("\nargmax 测试:")
    selector_argmax = PUCTSelector(c_puct=1.5, sample=False)
    for i in range(3):
        selected = selector_argmax.select(root)
        print(f"  {i + 1}. 选中: {selected.decision_action}")

    print("\nselect_path 测试:")
    selector2 = PUCTSelector(c_puct=1.5, sample=True)
    path = selector2.select_path(root, can_expand_fn=lambda n: True)
    print(f"  path 长度: {len(path)}")
    for i, node in enumerate(path):
        print(f"    [{i}] {node.state.sequence}")

    print("\n" + "=" * 60)
    print("测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
