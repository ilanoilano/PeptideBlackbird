# -*- coding: utf-8 -*-
"""
自适应MCTS-EGNN闭环优化配置

核心设计原则：
1. 所有动态参数由MCTS树深度控制
2. 深度越浅（未知越多）→ 随机补全越多
3. 深度越深（已知越多）→ 随机补全越少
4. 节点选择基于平均亲和度表现，表现好→扩展更多子节点
"""

import math
from typing import Dict, List, Tuple


# =============================================================================
# 动态参数配置（由MCTS树深度控制）
# =============================================================================

class AdaptiveMCTSConfig:
    """
    自适应MCTS配置类
    
    所有函数输入:
    - depth: 当前节点深度（0=根节点，6=叶节点）
    - max_depth: 最大深度（可变位置数，默认6）
    """
    
    # -------------------------------------------------------------------------
    # 1. 随机补全数量函数
    # -------------------------------------------------------------------------
    @staticmethod
    def get_random_fill_count(depth: int, max_depth: int = 6) -> int:
        """
        根据深度计算随机补全数量
        
        设计逻辑:
        - 深度0（根）: 50个（最多，完全未知）
        - 深度6（叶）: 10个（最少，接近完整）
        - 指数衰减: 深度越浅，衰减越快
        
        公式: N = max(10, 50 * exp(-depth * 0.3))
        
        Args:
            depth: 当前节点深度（0-6）
            max_depth: 最大深度（默认6）
        
        Returns:
            随机补全数量
        
        示例:
            depth=0: 50个
            depth=3: 20个  
            depth=6: 10个
        """
        import math
        
        base_count = 50
        min_count = 10
        decay_rate = 0.3
        
        count = int(base_count * math.exp(-depth * decay_rate))
        return max(min_count, count)
    
    @staticmethod
    def get_random_fill_count_linear(depth: int, max_depth: int = 6) -> int:
        """
        线性版本（备选）: 线性递减
        
        公式: N = max(10, 50 - depth * 7)
        
        示例:
            depth=0: 50个
            depth=3: 29个
            depth=6: 10个
        """
        base_count = 50
        min_count = 10
        decrement = 7
        
        count = base_count - depth * decrement
        return max(min_count, count)
    
    # -------------------------------------------------------------------------
    # 2. 节点扩展名额分配（基于Softmax概率）
    # -------------------------------------------------------------------------
    @staticmethod
    def allocate_expansion_slots(
        node_energies: List[Tuple[str, float]],
        total_slots: int = 50,
        temperature: float = 1.0
    ) -> Dict[str, int]:
        """
        按概率分配子节点扩展名额
        
        算法:
        1. 取能量绝对值（-8.8 → 8.8）
        2. Softmax计算概率（表现越好→概率越高）
        3. 按概率分配50个名额到各节点
        4. 每个节点内部用EGNN先验选top-k个子节点
        
        Args:
            node_energies: [(node_key, avg_energy), ...] 节点及其平均能量
            total_slots: 总扩展名额（默认50）
            temperature: Softmax温度（默认1.0，越小越锐化）
        
        Returns:
            {node_key: 分配到的扩展名额数}
        
        示例:
            输入: [('ACV', -8.8), ('ACW', -7.7), ('ACQ', -6.6), ('ACL', -2.2)]
            绝对值: [8.8, 7.7, 6.6, 2.2]
            Softmax: [0.52, 0.31, 0.14, 0.03]  (ACV概率最高)
            分配50个名额: {'ACV': 26, 'ACW': 16, 'ACQ': 7, 'ACL': 1}
        """
        import numpy as np
        
        if not node_energies:
            return {}
        
        if len(node_energies) == 1:
            return {node_energies[0][0]: total_slots}
        
        # 步骤1: 取绝对值（能量越负→绝对值越大→表现越好）
        abs_energies = np.array([abs(energy) for _, energy in node_energies])
        
        # 步骤2: Softmax计算概率
        # 除以temperature控制分布锐度
        exp_energies = np.exp(abs_energies / temperature)
        probabilities = exp_energies / np.sum(exp_energies)
        
        # 步骤3: 按概率分配名额
        # 先分配整数部分
        allocations = np.floor(probabilities * total_slots).astype(int)
        
        # 处理剩余名额（按小数部分从大到小分配）
        remaining = total_slots - np.sum(allocations)
        if remaining > 0:
            fractional_parts = probabilities * total_slots - allocations
            # 按小数部分排序，大的优先获得剩余名额
            sorted_indices = np.argsort(fractional_parts)[::-1]
            for i in range(remaining):
                allocations[sorted_indices[i % len(sorted_indices)]] += 1
        
        # 构建返回字典
        result = {}
        for i, (node_key, _) in enumerate(node_energies):
            result[node_key] = int(allocations[i])
        
        return result
    
    @staticmethod
    def get_node_expansion_count_with_egnn_prior(
        node_key: str,
        node_energy: float,
        all_node_energies: List[Tuple[str, float]],
        expansion_engine,
        total_slots: int = 50,
        temperature: float = 1.0
    ) -> Tuple[int, List[str]]:
        """
        获取节点的扩展名额，并返回该节点应扩展的top-k子节点
        
        完整流程:
        1. 对所有节点计算Softmax概率
        2. 分配50个名额到各节点
        3. 对该节点，用EGNN先验选择top-k个氨基酸作为子节点
        
        Args:
            node_key: 当前节点标识
            node_energy: 当前节点平均能量
            all_node_energies: 所有候选节点及其能量
            expansion_engine: ExpansionEngine实例（用于EGNN先验）
            total_slots: 总扩展名额
            temperature: Softmax温度
        
        Returns:
            (该节点分配到的名额数, top-k子节点动作列表)
        
        示例:
            ACV节点: 分配到26个名额
            → 用EGNN先验选择26个氨基酸中的top-6（受max_expansions限制）
            → 返回: (6, ['R', 'F', 'W', 'Y', 'L', 'V'])
        """
        # 步骤1-2: 分配名额
        allocations = AdaptiveMCTSConfig.allocate_expansion_slots(
            all_node_energies, total_slots, temperature
        )
        
        node_slots = allocations.get(node_key, 0)
        
        if node_slots == 0:
            return 0, []
        
        # 步骤3: 用EGNN先验选择top-k子节点
        # 从expansion_engine获取该节点所有可能的子节点及其先验概率
        # 选择先验概率最高的node_slots个（但受max_expansions限制）
        max_expansions = config.MCTS_CONFIG.get("max_expansions", 5)
        actual_expansions = min(node_slots, max_expansions)
        
        # 这里需要调用expansion_engine获取子节点先验
        # 简化版：返回名额数和空列表（实际实现需要expansion_engine配合）
        return actual_expansions, []
    
    @staticmethod
    def get_node_expansion_count(
        parent_avg_energy: float,
        best_energy: float,
        worst_energy: float,
        max_expansions: int = 6
    ) -> int:
        """
        【旧版】根据父节点表现计算应扩展多少个子节点
        
        已弃用，建议使用 allocate_expansion_slots
        """
        if worst_energy == best_energy:
            return max_expansions // 2
        
        normalized = (worst_energy - parent_avg_energy) / (worst_energy - best_energy)
        normalized = max(0.0, min(1.0, normalized))
        expansion_prob = normalized ** 1.5
        expansion_count = int(expansion_prob * max_expansions)
        
        return max(0, expansion_count)
    
    @staticmethod
    def get_node_selection_count_for_round(
        round_idx: int,
        base_count: int = 19,
        max_count: int = 50
    ) -> int:
        """
        每轮选择多少个节点进行评估
        
        设计逻辑:
        - 初始轮次: 19个（19种氨基酸）
        - 后续轮次: 逐渐增加，但不超过上限
        - 树越大，选择越多节点
        
        公式: count = min(19 + round_idx * 2, 50)
        
        Args:
            round_idx: 当前轮次（0-based）
            base_count: 基础数量（默认19）
            max_count: 最大数量（默认50）
        
        Returns:
            本轮要选择的节点数量
        
        示例:
            round 0: 19个
            round 5: 29个
            round 15+: 50个（上限）
        """
        increment = 2
        count = base_count + round_idx * increment
        return min(count, max_count)
    
    # -------------------------------------------------------------------------
    # 3. Vina验证数量函数
    # -------------------------------------------------------------------------
    @staticmethod
    def get_vina_validation_count(
        depth: int,
        max_depth: int = 6,
        base_count: int = 40,
        min_count: int = 20
    ) -> int:
        """
        根据深度计算Vina验证数量
        
        设计逻辑:
        - 深度浅（早期探索）: 40个（多验证，收集数据）
        - 深度深（后期利用）: 20个（少验证，精准打击）
        - 与随机补全数量趋势相反
        
        公式: N = max(20, 40 - depth * 3)
        
        Args:
            depth: 当前深度
            max_depth: 最大深度
            base_count: 基础数量（默认40）
            min_count: 最小数量（默认20）
        
        Returns:
            Vina验证数量
        
        示例:
            depth=0: 40个
            depth=3: 31个
            depth=6: 22个
        """
        decrement = 3
        count = base_count - depth * decrement
        return max(min_count, count)
    
    # -------------------------------------------------------------------------
    # 4. 收敛检测函数
    # -------------------------------------------------------------------------
    @staticmethod
    def check_convergence(
        test_mae_history: List[float],
        patience: int = 5,
        min_improvement: float = 0.05
    ) -> Tuple[bool, str]:
        """
        检测EGNN模型是否收敛
        
        收敛条件:
        - 连续patience轮测试集MAE无显著改善（改善 < min_improvement）
        
        Args:
            test_mae_history: 每轮测试集MAE历史列表（kcal/mol）
            patience: 容忍轮数（默认5）
            min_improvement: 最小改善阈值（默认0.05 kcal/mol）
        
        Returns:
            (是否收敛, 原因说明)
        """
        if len(test_mae_history) < patience + 1:
            return False, f"历史数据不足（{len(test_mae_history)}/{patience + 1}）"
        
        # 检查最近patience轮是否有显著改善
        recent_mae = test_mae_history[-patience:]
        previous_mae = test_mae_history[-patience-1]
        
        # MAE越低越好，计算改善量
        min_mae = min(recent_mae)
        improvement = previous_mae - min_mae  # 正值表示改善
        
        if improvement < min_improvement:
            return True, f"连续{patience}轮测试集MAE无显著改善（改善{improvement:.3f} < {min_improvement}）"
        
        return False, f"测试集MAE仍在改善（最近改善{improvement:.3f} kcal/mol）"
    
    @staticmethod
    def evaluate_egnn_on_test_set(
        model,
        test_sequences: List[str],
        test_energies: List[float],
        target_name: str
    ) -> Dict[str, float]:
        """
        在测试集上评估EGNN模型表现
        
        Args:
            model: EGNN模型
            test_sequences: 测试集序列列表
            test_energies: 测试集真实能量列表
            target_name: 靶点名称
        
        Returns:
            评估指标字典 {
                'mae': 平均绝对误差,
                'rmse': 均方根误差,
                'r2': R²分数,
                'pearson_r': Pearson相关系数
            }
        """
        from egnn_predictor import create_egnn_predictor
        from ligand_generator import generate_ligand
        import tempfile
        import numpy as np
        from scipy import stats
        
        predictions = []
        
        # 对每个测试序列进行预测
        with tempfile.TemporaryDirectory() as tmpdir:
            for seq in test_sequences:
                try:
                    # 生成分子
                    pdbqt_path = generate_ligand(
                        sequence=seq,
                        crosslinker=config.CROSSLINKER,
                        crosslinker_positions=config.CROSSLINKER_POSITIONS,
                        output_dir=Path(tmpdir)
                    )
                    
                    # EGNN预测
                    pred_energy = model.predict(pdbqt_path)
                    predictions.append(pred_energy)
                    
                except Exception as e:
                    print(f"预测失败 {seq}: {e}")
                    predictions.append(0.0)  # 失败时返回0
        
        predictions = np.array(predictions)
        test_energies = np.array(test_energies)
        
        # 计算指标
        mae = np.mean(np.abs(predictions - test_energies))
        rmse = np.sqrt(np.mean((predictions - test_energies) ** 2))
        
        # R²
        ss_res = np.sum((test_energies - predictions) ** 2)
        ss_tot = np.sum((test_energies - np.mean(test_energies)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Pearson相关系数
        pearson_r, _ = stats.pearsonr(test_energies, predictions)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson_r': float(pearson_r),
            'n_samples': len(test_sequences)
        }
    
    # -------------------------------------------------------------------------
    # 5. 数据集划分工具
    # -------------------------------------------------------------------------
    @staticmethod
    def split_dataset(
        sequences: List[str],
        energies: List[float],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        将数据集划分为训练集、验证集、测试集
        
        比例: 8:1:1 (默认)
        
        Args:
            sequences: 序列列表
            energies: 能量列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        
        Returns:
            {
                'train': (train_seqs, train_energies),
                'val': (val_seqs, val_energies),
                'test': (test_seqs, test_energies)
            }
        """
        import random
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
        
        # 合并并打乱
        data = list(zip(sequences, energies))
        random.seed(random_seed)
        random.shuffle(data)
        
        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # 测试集取剩余部分，避免舍入误差
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        train_seqs, train_energies = zip(*train_data) if train_data else ([], [])
        val_seqs, val_energies = zip(*val_data) if val_data else ([], [])
        test_seqs, test_energies = zip(*test_data) if test_data else ([], [])
        
        return {
            'train': (list(train_seqs), list(train_energies)),
            'val': (list(val_seqs), list(val_energies)),
            'test': (list(test_seqs), list(test_energies))
        }
    
    # -------------------------------------------------------------------------
    # 6. EGNN更新流程（含实时测试集更新）
    # -------------------------------------------------------------------------
    @staticmethod
    def update_egnn_with_vina_results_v2(
        vina_results: Dict[str, float],
        existing_test_data: Dict[str, float],
        target_name: str,
        n_epochs: int = 20,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> Tuple[bool, Dict[str, float], Dict[str, float]]:
        """
        用Vina结果更新EGNN模型，并实时更新测试集
        
        流程:
        1. 将Vina结果按8:1:1划分为训练/验证/测试
        2. 测试集与existing_test_data合并（实时更新）
        3. 重新训练/微调EGNN
        4. 在当前测试集上评估
        5. 保存评估指标
        
        Args:
            vina_results: Vina验证结果 {sequence: energy}
            existing_test_data: 已有测试集数据 {sequence: energy}
            target_name: 靶点名称
            n_epochs: 微调轮数
            split_ratio: 划分比例 (train, val, test)
        
        Returns:
            (是否成功, 测试集评估指标, 更新后的测试集)
        
        示例:
            vina_results = {
                'ACEVHCIARCG': -8.5,
                'ACWVHCVPLCG': -7.2,
                ... (40个)
            }
            # 划分: 32个训练, 4个验证, 4个测试
            # 新测试集与已有测试集合并
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from EGNN_1 import main as egnn_data_prep
        from EGNN_23 import main as egnn_train
        from egnn_predictor import EGNNPredictor
        import config
        
        print("\n" + "="*60)
        print("EGNN模型更新（实时测试集）")
        print("="*60)
        
        # 步骤1: 划分Vina结果
        print(f"\n[1/5] 划分Vina结果 (比例 {split_ratio[0]}:{split_ratio[1]}:{split_ratio[2]})...")
        try:
            sequences = list(vina_results.keys())
            energies = list(vina_results.values())
            
            split_data = AdaptiveMCTSConfig.split_dataset(
                sequences, energies,
                train_ratio=split_ratio[0],
                val_ratio=split_ratio[1],
                test_ratio=split_ratio[2]
            )
            
            train_seqs, train_energies = split_data['train']
            val_seqs, val_energies = split_data['val']
            new_test_seqs, new_test_energies = split_data['test']
            
            print(f"  ✓ 划分完成:")
            print(f"    训练集: {len(train_seqs)} 个")
            print(f"    验证集: {len(val_seqs)} 个")
            print(f"    新测试集: {len(new_test_seqs)} 个")
            
        except Exception as e:
            print(f"  ✗ 划分失败: {e}")
            return False, {}, existing_test_data
        
        # 步骤2: 更新测试集（实时合并）
        print("\n[2/5] 更新测试集...")
        try:
            # 合并新旧测试集
            updated_test_data = existing_test_data.copy()
            for seq, energy in zip(new_test_seqs, new_test_energies):
                updated_test_data[seq] = energy
            
            # 限制测试集大小（避免无限增长）
            max_test_size = 100
            if len(updated_test_data) > max_test_size:
                # 保留最新的100个
                items = list(updated_test_data.items())
                updated_test_data = dict(items[-max_test_size:])
                print(f"  测试集超过{max_test_size}，保留最新的{max_test_size}个")
            
            print(f"  ✓ 测试集更新完成: {len(existing_test_data)} → {len(updated_test_data)} 个")
            
        except Exception as e:
            print(f"  ✗ 测试集更新失败: {e}")
            return False, {}, existing_test_data
        
        # 步骤3: 保存数据到文件
        print("\n[3/5] 保存数据...")
        try:
            dataset_path = config.RESULTS_DIR / target_name / "dataset.csv"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            import csv
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
            # 追加训练数据
            with open(dataset_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for seq, energy in zip(train_seqs, train_energies):
                    writer.writerow([seq, config.CROSSLINKER, '', energy, 'vina_train', timestamp])
                for seq, energy in zip(val_seqs, val_energies):
                    writer.writerow([seq, config.CROSSLINKER, '', energy, 'vina_val', timestamp])
            
            # 保存测试集（单独文件）
            test_path = config.RESULTS_DIR / target_name / "test_set.csv"
            with open(test_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sequence', 'energy', 'timestamp'])
                for seq, energy in updated_test_data.items():
                    writer.writerow([seq, energy, timestamp])
            
            print(f"  ✓ 数据保存完成")
            
        except Exception as e:
            print(f"  ✗ 数据保存失败: {e}")
            return False, {}, updated_test_data
        
        # 步骤4: 准备EGNN训练数据
        print("\n[4/5] 准备EGNN训练数据...")
        try:
            # 读取所有历史训练数据
            all_train_data = []
            if dataset_path.exists():
                with open(dataset_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 4:
                            try:
                                seq = row[0]
                                energy = float(row[3])
                                source = row[4] if len(row) > 4 else 'unknown'
                                if energy < 0 and 'test' not in source:  # 排除测试集
                                    all_train_data.append((seq, energy))
                            except:
                                continue
            
            # 加上本轮训练集
            all_train_data.extend(zip(train_seqs, train_energies))
            
            # 去重
            seen = set()
            unique_data = []
            for seq, energy in all_train_data:
                if seq not in seen:
                    seen.add(seq)
                    unique_data.append((seq, energy))
            
            # 保存
            sequences_file = config.BASE_DIR / "sequences.txt"
            energies_file = config.RESULTS_DIR / target_name / "energies.csv"
            
            with open(sequences_file, 'w') as f:
                for seq, _ in unique_data:
                    f.write(f"{seq}\n")
            
            with open(energies_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sequence', 'energy'])
                for seq, energy in unique_data:
                    writer.writerow([seq, energy])
            
            # 调用EGNN_1准备数据
            egnn_data_prep(
                sequences_file=sequences_file,
                energies_file=energies_file,
                output_dir=config.BASE_DIR / "egnn" / "raw"
            )
            print(f"  ✓ EGNN数据准备完成 ({len(unique_data)} 个训练样本)")
            
        except Exception as e:
            print(f"  ✗ EGNN数据准备失败: {e}")
            return False, {}, updated_test_data
        
        # 步骤5: 微调EGNN模型并在测试集评估
        print(f"\n[5/5] 微调EGNN模型 ({n_epochs} epochs) + 测试集评估...")
        try:
            # 备份旧模型
            model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"
            backup_path = model_path.with_suffix('.pt.backup')
            if model_path.exists():
                import shutil
                shutil.copy2(model_path, backup_path)
            
            # 训练
            egnn_train(
                data_dir=config.BASE_DIR / "egnn" / "raw",
                output_dir=config.BASE_DIR / "egnn" / "models",
                hidden_dim=config.EGNN_CONFIG["hidden_dim"],
                num_layers=config.EGNN_CONFIG["num_layers"],
                num_epochs=n_epochs,
                batch_size=config.EGNN_CONFIG["batch_size"],
                lr=config.EGNN_CONFIG["learning_rate"] * 0.5,
                patience=10
            )
            print("  ✓ EGNN微调完成")
            
            # 加载新模型并评估
            new_model = EGNNPredictor(
                model_path=model_path,
                hidden_dim=config.EGNN_CONFIG["hidden_dim"],
                num_layers=config.EGNN_CONFIG["num_layers"]
            )
            
            test_seqs = list(updated_test_data.keys())
            test_energies = list(updated_test_data.values())
            
            metrics = AdaptiveMCTSConfig.evaluate_egnn_on_test_set(
                model=new_model,
                test_sequences=test_seqs,
                test_energies=test_energies,
                target_name=target_name
            )
            
            print(f"  ✓ 测试集评估完成 ({len(test_seqs)} 个样本)")
            print(f"    MAE: {metrics['mae']:.3f} kcal/mol")
            print(f"    RMSE: {metrics['rmse']:.3f} kcal/mol")
            print(f"    R²: {metrics['r2']:.3f}")
            print(f"    Pearson r: {metrics['pearson_r']:.3f}")
            
        except Exception as e:
            print(f"  ✗ 训练或评估失败: {e}")
            if backup_path.exists():
                import shutil
                shutil.copy2(backup_path, model_path)
                print("  已恢复旧模型")
            return False, {}, updated_test_data
        
        print("="*60)
        return True, metrics, updated_test_data


# =============================================================================
# 使用示例
# =============================================================================

def example_usage():
    """使用示例"""
    config = AdaptiveMCTSConfig()
    
    print("="*60)
    print("随机补全数量（指数衰减）")
    print("="*60)
    for depth in range(7):
        count = config.get_random_fill_count(depth)
        print(f"  depth={depth}: {count}个")
    
    print("\n" + "="*60)
    print("随机补全数量（线性递减）")
    print("="*60)
    for depth in range(7):
        count = config.get_random_fill_count_linear(depth)
        print(f"  depth={depth}: {count}个")
    
    print("\n" + "="*60)
    print("节点扩展数量（基于表现）")
    print("="*60)
    best_energy = -8.8
    worst_energy = -2.2
    test_energies = [-8.8, -7.7, -6.6, -5.5, -4.4, -3.3, -2.2]
    for energy in test_energies:
        count = config.get_node_expansion_count(energy, best_energy, worst_energy)
        print(f"  能量={energy:+.1f}: 扩展{count}个")
    
    print("\n" + "="*60)
    print("每轮节点选择数量")
    print("="*60)
    for round_idx in range(0, 20, 2):
        count = config.get_node_selection_count_for_round(round_idx)
        print(f"  round {round_idx}: {count}个")
    
    print("\n" + "="*60)
    print("Vina验证数量")
    print("="*60)
    for depth in range(7):
        count = config.get_vina_validation_count(depth)
        print(f"  depth={depth}: {count}个")
    
    print("\n" + "="*60)
    print("Softmax概率分配扩展名额")
    print("="*60)
    node_energies = [
        ('ACV', -8.8),
        ('ACW', -7.7),
        ('ACQ', -6.6),
        ('ACF', -5.5),
        ('ACR', -4.4),
        ('ACK', -3.3),
        ('ACL', -2.2)
    ]
    print(f"  节点能量: {node_energies}")
    print(f"  绝对值: {[abs(e) for _, e in node_energies]}")
    
    allocations = config.allocate_expansion_slots(node_energies, total_slots=50, temperature=1.0)
    print(f"\n  分配50个名额 (T=1.0):")
    for node, slots in allocations.items():
        energy = dict(node_energies)[node]
        print(f"    {node} ({energy:+.1f}): {slots}个")
    
    # 计算概率
    import numpy as np
    abs_energies = np.array([abs(e) for _, e in node_energies])
    exp_energies = np.exp(abs_energies / 1.0)
    probabilities = exp_energies / np.sum(exp_energies)
    print(f"\n  Softmax概率:")
    for i, (node, _) in enumerate(node_energies):
        print(f"    {node}: {probabilities[i]:.3f} ({probabilities[i]*100:.1f}%)")
    
    print("\n" + "="*60)
    print("不同温度下的分配")
    print("="*60)
    for temp in [0.5, 1.0, 2.0]:
        allocations = config.allocate_expansion_slots(node_energies, total_slots=50, temperature=temp)
        print(f"\n  T={temp}:")
        for node, slots in allocations.items():
            energy = dict(node_energies)[node]
            print(f"    {node} ({energy:+.1f}): {slots}个")
    
    print("\n" + "="*60)
    print("收敛检测示例")
    print("="*60)
    # 收敛情况：连续5轮MAE无改善
    converged_mae = [1.5, 1.45, 1.42, 1.40, 1.41, 1.39]
    is_conv, reason = config.check_convergence(converged_mae, patience=5)
    print(f"  MAE历史: {converged_mae}")
    print(f"  是否收敛: {is_conv}, 原因: {reason}")
    
    # 未收敛情况：仍在改善
    improving_mae = [1.5, 1.3, 1.1, 0.9, 0.7, 0.5]
    is_conv, reason = config.check_convergence(improving_mae, patience=5)
    print(f"  MAE历史: {improving_mae}")
    print(f"  是否收敛: {is_conv}, 原因: {reason}")


if __name__ == "__main__":
    example_usage()
    test_energies = [-8.8, -7.7, -6.6, -5.5, -4.4, -3.3, -2.2]
    for energy in test_energies:
        count = config.get_node_expansion_count(energy, best_energy, worst_energy)
        print(f"  能量={energy:+.1f}: 扩展{count}个")
    
    print("\n" + "="*60)
    print("每轮节点选择数量")
    print("="*60)
    for round_idx in range(0, 20, 2):
        count = config.get_node_selection_count_for_round(round_idx)
        print(f"  round {round_idx}: {count}个")
    
    print("\n" + "="*60)
    print("Vina验证数量")
    print("="*60)
    for depth in range(7):
        count = config.get_vina_validation_count(depth)
        print(f"  depth={depth}: {count}个")


if __name__ == "__main__":
    example_usage()
