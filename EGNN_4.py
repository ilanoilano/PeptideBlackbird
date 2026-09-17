#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGNN评估模块 (EGNN_4.py)
功能：模型评估 + 可视化

输入：
- egnn/models/best_model.pt
- egnn/raw/test_data.npz

处理逻辑：
1. 加载训练好的模型
2. 在测试集上做预测
3. 计算整体指标：R², MAE, RMSE
4. 分组MAE柱状图：
   - 按预测分数排序，等分10组
   - 每组计算MAE
   - 绘制柱状图
5. 生成评估报告

输出：
- egnn/evaluate/evaluation_report.txt
- egnn/evaluate/grouped_mae.png
- egnn/evaluate/predictions_vs_true.png
"""

import os
import sys
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from vina import get_pocket_center
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))

import config
from EGNN_23 import EGNNModel, collate_graphs, load_npz_data


# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path: Path, hidden_dim: int = 128, num_layers: int = 4):
    """加载训练好的模型（兼容 torch.compile 保存的权重）"""
    model = EGNNModel(in_features=20, hidden_dim=hidden_dim, num_layers=num_layers)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict']

    # 【修复】移除 _orig_mod. 前缀（如果存在）
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[10:]  # 去掉 '_orig_mod.' 前缀
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


def predict(model, test_data, batch_size: int = 8):
    """
    在测试集上做预测
    
    Returns:
        y_true: 真实值
        y_pred: 预测值
    """
    y_true = []
    y_pred = []
    
    # 分批处理
    test_loader = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
    
    with torch.no_grad():
        for batch_data in test_loader:
            features, coords, edge_index, batch, energies = collate_graphs(batch_data)
            
            features = features.to(DEVICE)
            coords = coords.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            batch = batch.to(DEVICE)
            
            pred_energies = model(features, coords, edge_index, batch)
            
            y_true.extend(energies.cpu().numpy())
            y_pred.extend(pred_energies.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    计算评估指标
    
    Returns:
        {
            'r2': R²,
            'mae': MAE,
            'rmse': RMSE
        }
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }


def calculate_grouped_mae(y_true: np.ndarray, y_pred: np.ndarray, n_groups: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算分组MAE

    按预测分数排序，等分n_groups组，每组计算MAE
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print("警告: 测试集为空，跳过分组MAE计算")
        return np.array([]), np.array([])

    # 确保数据是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 【新增】如果样本数少于分组数，减少分组数
    n_samples = len(y_true)
    if n_samples < n_groups:
        n_groups = max(1, n_samples // 2)  # 至少2个样本一组，或直接减半
        print(f"  样本数({n_samples})少于分组数，自动调整为 {n_groups} 组")

    # 按预测值排序
    sorted_indices = np.argsort(y_pred)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # 等分
    group_size = n_samples // n_groups

    percentiles = []
    group_maes = []

    for i in range(n_groups):
        start = i * group_size
        end = (i + 1) * group_size if i < n_groups - 1 else n_samples

        y_true_group = y_true_sorted[start:end]
        y_pred_group = y_pred_sorted[start:end]

        # 【新增】跳过空组
        if len(y_true_group) == 0:
            continue

        # 计算MAE
        mae = mean_absolute_error(y_true_group, y_pred_group)
        group_maes.append(mae)

        # 百分位数（组中点）
        percentile = (i + 0.5) * 100 / n_groups
        percentiles.append(percentile)

    return np.array(percentiles), np.array(group_maes)


def plot_grouped_mae(percentiles: np.ndarray, group_maes: np.ndarray, 
                     output_path: Path, metrics: Dict):
    """
    绘制分组MAE柱状图
    
    横轴：百分位数区间（0-10%, 10-20%, ..., 90-100%）
    纵轴：该区间的MAE
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 柱状图
    bars = ax.bar(percentiles, group_maes, width=8, alpha=0.7, color='steelblue', edgecolor='black')
    
    # 高亮第一个柱子（低分区间，最重要的区域）
    bars[0].set_color('coral')
    bars[0].set_label('Lowest 10% (Best candidates)')
    
    # 添加数值标签
    for i, (p, mae) in enumerate(zip(percentiles, group_maes)):
        ax.text(p, mae + 0.02, f'{mae:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 添加平均线
    avg_mae = np.mean(group_maes)
    ax.axhline(y=avg_mae, color='red', linestyle='--', label=f'Average MAE: {avg_mae:.3f}')
    
    # 设置标签
    ax.set_xlabel('Prediction Percentile (%)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (kcal/mol)', fontsize=12)
    ax.set_title(f'Grouped MAE by Prediction Percentile\nR²={metrics["r2"]:.3f}, MAE={metrics["mae"]:.3f}, RMSE={metrics["rmse"]:.3f}', 
                 fontsize=14)
    
    # 设置x轴刻度
    ax.set_xticks(percentiles)
    ax.set_xticklabels([f'{int(p-5)}-{int(p+5)}%' for p in percentiles], rotation=45, ha='right')
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存分组MAE图: {output_path}")


def plot_predictions_vs_true(y_true: np.ndarray, y_pred: np.ndarray, 
                              output_path: Path, metrics: Dict):
    """
    绘制预测值vs真实值散点图
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # 理想线 y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    
    # 设置标签
    ax.set_xlabel('True Energy (kcal/mol)', fontsize=12)
    ax.set_ylabel('Predicted Energy (kcal/mol)', fontsize=12)
    ax.set_title(f'Predictions vs True Values\nR²={metrics["r2"]:.3f}, MAE={metrics["mae"]:.3f}, RMSE={metrics["rmse"]:.3f}', 
                 fontsize=14)
    
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存预测vs真实图: {output_path}")


def generate_report(metrics: Dict, percentiles: np.ndarray, group_maes: np.ndarray,
                    output_path: Path):
    """
    生成评估报告
    """
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EGNN模型评估报告\n")
        f.write("="*60 + "\n\n")
        
        # 整体指标
        f.write("【整体指标】\n")
        f.write(f"R² (决定系数): {metrics['r2']:.4f}\n")
        f.write(f"MAE (平均绝对误差): {metrics['mae']:.4f} kcal/mol\n")
        f.write(f"RMSE (均方根误差): {metrics['rmse']:.4f} kcal/mol\n\n")
        
        # 分组MAE
        f.write("【分组MAE】\n")
        f.write("百分位区间 | MAE (kcal/mol)\n")
        f.write("-"*40 + "\n")
        
        for p, mae in zip(percentiles, group_maes):
            start = int(p - 5)
            end = int(p + 5)
            marker = " <-- 重点关注" if start == 0 else ""
            f.write(f"{start:3d}-{end:3d}%   | {mae:.4f}{marker}\n")
        
        f.write("\n")
        
        # 解读
        f.write("【结果解读】\n")
        
        # 最低10%区间的MAE
        lowest_mae = group_maes[0]
        avg_mae = np.mean(group_maes)
        
        if lowest_mae > avg_mae * 1.2:
            f.write("⚠️ 警告: 最低10%区间（最佳候选分子）的MAE显著高于平均水平，\n")
            f.write("   说明模型对优质分子的预测不够准确，建议：\n")
            f.write("   1. 增加训练数据中优质分子的比例\n")
            f.write("   2. 调整损失函数，增加对低分样本的权重\n")
            f.write("   3. 增加模型复杂度或训练轮数\n")
        elif lowest_mae < avg_mae * 0.8:
            f.write("✓ 良好: 最低10%区间的MAE低于平均水平，\n")
            f.write("   说明模型对优质分子的预测较为准确。\n")
        else:
            f.write("○ 一般: 最低10%区间的MAE与平均水平相当。\n")
        
        f.write("\n")
        
        # 整体评价
        if metrics['r2'] > 0.8:
            f.write("✓ 模型表现优秀 (R² > 0.8)\n")
        elif metrics['r2'] > 0.6:
            f.write("○ 模型表现良好 (R² > 0.6)\n")
        else:
            f.write("⚠️ 模型表现一般 (R² < 0.6)，建议优化\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ 保存评估报告: {output_path}")


def main(model_path: Optional[Path] = None,
         test_data_path: Optional[Path] = None,
         output_dir: Optional[Path] = None,
         hidden_dim: int = 128,
         num_layers: int = 4,
         target_name: Optional[str] = None):
    """
    主评估函数

    Args:
        target_name: 靶点名称（用于确定EGNN目录）
    """
    # 使用target_name确定EGNN目录
    if target_name is not None:
        egnn_dirs = config.get_egnn_dirs(target_name)
        if model_path is None:
            model_path = egnn_dirs["models"] / "best_model.pt"
        if test_data_path is None:
            test_data_path = egnn_dirs["raw"] / "test_data.npz"
        if output_dir is None:
            output_dir = egnn_dirs["evaluate"]
    else:
        # fallback：使用原默认路径
        if model_path is None:
            model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"
        if test_data_path is None:
            test_data_path = config.BASE_DIR / "egnn" / "raw" / "test_data.npz"
        if output_dir is None:
            output_dir = config.BASE_DIR / "egnn" / "evaluate"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EGNN模型评估")
    print("="*60)
    
    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    model = load_model(model_path, hidden_dim, num_layers)
    print(f"  模型: {model_path}")
    
    # 2. 加载测试数据
    print("\n[2/4] 加载测试数据...")
    test_data = load_npz_data(test_data_path)
    print(f"  测试集: {len(test_data)} 个样本")
    
    if len(test_data) == 0:
        print("错误: 测试集为空")
        return
    
    # 3. 预测
    print("\n[3/4] 进行预测...")
    y_true, y_pred = predict(model, test_data)
    print(f"  完成 {len(y_true)} 个样本的预测")
    
    # 4. 计算指标
    print("\n[4/4] 计算指标并生成可视化...")
    metrics = calculate_metrics(y_true, y_pred)
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f} kcal/mol")
    print(f"  RMSE: {metrics['rmse']:.4f} kcal/mol")
    
    # 5. 分组MAE
    percentiles, group_maes = calculate_grouped_mae(y_true, y_pred, n_groups=10)
    
    # 6. 生成可视化
    plot_grouped_mae(percentiles, group_maes, output_dir / "grouped_mae.png", metrics)
    plot_predictions_vs_true(y_true, y_pred, output_dir / "predictions_vs_true.png", metrics)
    
    # 7. 生成报告
    generate_report(metrics, percentiles, group_maes, output_dir / "evaluation_report.txt")
    
    print("\n" + "="*60)
    print("评估完成!")
    print(f"结果保存至: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EGNN模型评估')
    parser.add_argument('--model-path', type=Path, default=None)
    parser.add_argument('--test-data', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)
    
    parser.add_argument('--target', type=str, default=None, help='靶点名称（如 6Y9A）')
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        target_name=args.target
    )

def calculate_pocket_distances(sequences: List[str], target_name: Optional[str] = None) -> np.ndarray:
    """
    计算配体质心到口袋中心的距离

    Args:
        sequences: 序列列表
        target_name: 靶点名称
    """
    if target_name is None:
        target_name = "1LYZ"  # fallback


    from ligand_generator import generate_ligand
    import tempfile
    
    pocket_center = get_pocket_center(target_name)
    if pocket_center is None:
        print(f"【警告】无法获取口袋中心，返回空数组")
        return np.array([])
    
    distances = []
    for seq in sequences:
        try:
            # 生成配体
            with tempfile.TemporaryDirectory() as tmpdir:
                pdbqt_path = generate_ligand(
                    sequence=seq,
                    output_dir=Path(tmpdir),
                    random_seed=42
                )
                
                # 读取坐标
                coords = []
                with open(pdbqt_path, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            try:
                                x = float(line[30:38].strip())
                                y = float(line[38:46].strip())
                                z = float(line[46:54].strip())
                                coords.append([x, y, z])
                            except:
                                continue
                
                if len(coords) > 0:
                    coords = np.array(coords)
                    centroid = np.mean(coords, axis=0)
                    distance = np.linalg.norm(centroid - pocket_center)
                    distances.append(distance)
                else:
                    distances.append(np.nan)
        except Exception as e:
            print(f"【警告】计算 {seq} 的距离失败: {e}")
            distances.append(np.nan)
    
    return np.array(distances)


def plot_pocket_distance_histogram(distances: np.ndarray, output_path: Path = None):
    """
    绘制配体质心到口袋中心距离的直方图
    
    Args:
        distances: 距离数组（Å）
        output_path: 输出文件路径
    """
    # 过滤掉 NaN
    valid_distances = distances[~np.isnan(distances)]
    
    if len(valid_distances) == 0:
        print("【警告】没有有效的距离数据，跳过绘图")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 直方图
    plt.hist(valid_distances, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    
    # 统计信息
    mean_dist = np.mean(valid_distances)
    median_dist = np.median(valid_distances)
    std_dist = np.std(valid_distances)
    
    # 添加垂直线
    plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_dist:.2f}Å')
    plt.axvline(median_dist, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_dist:.2f}Å')
    
    # 添加阈值线
    plt.axvline(5.0, color='orange', linestyle=':', linewidth=2, label='阈值: 5.0Å')
    plt.axvline(7.0, color='purple', linestyle=':', linewidth=2, label='阈值: 7.0Å')
    
    plt.xlabel('质心到口袋中心的距离 (Å)', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title(f'配体质心偏离口袋中心的距离分布 (n={len(valid_distances)})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加统计文本



    stats_text = f'''均值: {mean_dist:.2f}Å
    中位数: {median_dist:.2f}Å
    标准差: {std_dist:.2f}Å
    最小: {np.min(valid_distances):.2f}Å
    最大: {np.max(valid_distances):.2f}Å'''
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 距离分布图已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_distance_vs_error(distances: np.ndarray, errors: np.ndarray, output_path: Path = None):
    """
    绘制距离 vs 预测误差的关系图
    
    Args:
        distances: 距离数组（Å）
        errors: 预测误差数组（kcal/mol）
        output_path: 输出文件路径
    """
    # 过滤掉 NaN
    mask = ~np.isnan(distances) & ~np.isnan(errors)
    valid_distances = distances[mask]
    valid_errors = errors[mask]
    
    if len(valid_distances) == 0:
        print("【警告】没有有效的数据，跳过绘图")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 散点图
    plt.scatter(valid_distances, valid_errors, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
    
    # 拟合线
    z = np.polyfit(valid_distances, valid_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_distances.min(), valid_distances.max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'线性拟合: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # 添加阈值线
    plt.axvline(5.0, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='阈值: 5.0Å')
    plt.axvline(7.0, color='purple', linestyle=':', linewidth=2, alpha=0.7, label='阈值: 7.0Å')
    plt.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.xlabel('质心到口袋中心的距离 (Å)', fontsize=12)
    plt.ylabel('预测误差 (kcal/mol)', fontsize=12)
    plt.title(f'距离 vs 预测误差 (n={len(valid_distances)}, r={np.corrcoef(valid_distances, valid_errors)[0,1]:.3f})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 距离-误差关系图已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_pocket_distance_report(distances: np.ndarray, output_dir: Path = None):
    """
    生成配体距离统计报告
    
    Args:
        distances: 距离数组（Å）
        output_dir: 输出目录
    """
    valid_distances = distances[~np.isnan(distances)]
    
    if len(valid_distances) == 0:
        print("【警告】没有有效的距离数据")
        return
    
    # 统计
    stats = {
        'count': len(valid_distances),
        'mean': float(np.mean(valid_distances)),
        'median': float(np.median(valid_distances)),
        'std': float(np.std(valid_distances)),
        'min': float(np.min(valid_distances)),
        'max': float(np.max(valid_distances)),
        'within_5A': int(np.sum(valid_distances <= 5.0)),
        'within_7A': int(np.sum(valid_distances <= 7.0)),
        'percent_within_5A': float(np.sum(valid_distances <= 5.0) / len(valid_distances) * 100),
        'percent_within_7A': float(np.sum(valid_distances <= 7.0) / len(valid_distances) * 100),
    }
    

    print("配体质心偏离口袋中心距离统计")
    print("="*60)
    print(f"样本数: {stats['count']}")
    print(f"均值: {stats['mean']:.2f}Å")
    print(f"中位数: {stats['median']:.2f}Å")
    print(f"标准差: {stats['std']:.2f}Å")
    print(f"最小值: {stats['min']:.2f}Å")
    print(f"最大值: {stats['max']:.2f}Å")
    print(f"在 5Å 内: {stats['within_5A']} ({stats['percent_within_5A']:.1f}%)")
    print(f"在 7Å 内: {stats['within_7A']} ({stats['percent_within_7A']:.1f}%)")
    print("="*60)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON
        json_path = output_dir / "pocket_distance_stats.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ 统计报告已保存: {json_path}")
        
        # 绘制图表
        plot_pocket_distance_histogram(distances, output_dir / "pocket_distance_histogram.png")
    
    return stats
