# -*- coding: utf-8 -*-
"""
MCTS 日志管理模块
功能：管理MCTS运行日志，支持批量写入和分级日志
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

import config


class MCTSLogger:
    """
    MCTS专用日志管理器
    
    特性：
    1. 控制台只输出简洁进度信息
    2. 详细调试信息批量写入日志文件
    3. 每100个分子批量写入一次
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, target_name: str = "default", log_dir: Optional[Path] = None):
        if self._initialized:
            return
        
        self.target_name = target_name
        self.log_dir = log_dir or (config.BASE_DIR / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"{target_name}_{timestamp}.log"
        self.debug_file = self.log_dir / f"{target_name}_{timestamp}_debug.log"
        
        # 统计信息
        self.stats = {
            "total_molecules": 0,
            "egnn_iterations": 0,
            "mcts_depth_max": 0,
            "random_molecules": 0,
            "start_time": time.time(),
        }
        
        # 批量写入缓冲区
        self.debug_buffer = deque(maxlen=100)
        self.buffer_lock = threading.Lock()
        
        # 写入计数器
        self.write_counter = 0
        
        # 初始化日志文件
        self._init_log_files()
        
        self._initialized = True
        print(f"[MCTSLogger] 日志初始化: {self.log_file}")
    
    def _init_log_files(self):
        """初始化日志文件"""
        header = f"""# MCTS Log - {self.target_name}
# Started: {datetime.now().isoformat()}
# ========================================

"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
        
        with open(self.debug_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def log_progress(self, egnn_iter: int, mcts_depth: int, random_count: int, 
                     sequence: str = "", energy: float = None, 
                     max_iterations: int = None):
        """
        输出简洁的进度信息到控制台
        
        格式: 【EGNN迭代轮次 X/MAX | MCTS深度: Y | 已生成随机分子: Z】
        
        Args:
            egnn_iter: 当前EGNN迭代轮次
            mcts_depth: MCTS当前深度
            random_count: 已生成随机分子数
            sequence: 当前序列（可选）
            energy: 预测能量（可选）
            max_iterations: 最大迭代轮次（可选，默认从config读取）
        """
        # 更新统计
        self.stats["egnn_iterations"] = egnn_iter
        self.stats["mcts_depth_max"] = max(self.stats["mcts_depth_max"], mcts_depth)
        self.stats["random_molecules"] = random_count
        
        # 确定最大迭代次数
        if max_iterations is None:
            max_iterations = config.MAX_EGNN_ITERATIONS
        
        # 控制台输出简洁格式
        progress_msg = f"【EGNN迭代轮次 {egnn_iter}/{max_iterations} | MCTS深度: {mcts_depth} | 已生成随机分子: {random_count}"
        if sequence:
            progress_msg += f" | 序列: {sequence}"
        if energy is not None:
            progress_msg += f" | 能量: {energy:.4f}"
        progress_msg += "】"
        
        print(progress_msg)
        
        # 同时写入主日志
        self._write_to_log(progress_msg)
    
    def log_debug(self, component: str, message: str, data: Optional[Dict] = None):
        """
        记录调试信息（批量写入，不输出到控制台）
        
        Args:
            component: 组件名称（如 'ligand_generator', 'simulation'）
            message: 调试消息
            data: 额外数据字典
        """
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        log_entry = {
            "timestamp": timestamp,
            "component": component,
            "message": message,
            "data": data or {}
        }
        
        with self.buffer_lock:
            self.debug_buffer.append(log_entry)
            self.write_counter += 1
            
            # 每100条写入一次
            if self.write_counter >= 100:
                self._flush_debug_buffer()
    
    def _flush_debug_buffer(self):
        """将缓冲区内容写入调试日志文件"""
        if not self.debug_buffer:
            return
        
        lines = []
        while self.debug_buffer:
            entry = self.debug_buffer.popleft()
            line = f"[{entry['timestamp']}] [{entry['component']}] {entry['message']}"
            if entry['data']:
                line += f" | data: {json.dumps(entry['data'], ensure_ascii=False)}"
            lines.append(line)
        
        with open(self.debug_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        
        self.write_counter = 0
    
    def _write_to_log(self, message: str):
        """直接写入主日志文件"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_crosslinker_debug(self, cys_sulfur_indices: List[int], 
                              selected_sulfur_indices: List[int],
                              br_indices: List[int],
                              bonds_created: int,
                              bond_details: List[Dict] = None):
        """
        记录交联剂调试信息
        """
        data = {
            "cys_sulfur_indices": cys_sulfur_indices,
            "selected_sulfur_indices": selected_sulfur_indices,
            "br_indices": br_indices,
            "bonds_created": bonds_created,
            "bond_details": bond_details or []
        }
        self.log_debug("ligand_generator", "交联剂构建详情", data)
    
    def log_molecule_generated(self, sequence: str, crosslinker: str, 
                               energy: float, generation_time: float):
        """
        记录分子生成完成
        """
        self.stats["total_molecules"] += 1
        
        data = {
            "sequence": sequence,
            "crosslinker": crosslinker,
            "energy": energy,
            "generation_time_ms": round(generation_time * 1000, 2),
            "total_count": self.stats["total_molecules"]
        }
        self.log_debug("simulation", "分子生成完成", data)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        elapsed = time.time() - self.stats["start_time"]
        stats = self.stats.copy()
        stats["elapsed_time"] = round(elapsed, 2)
        stats["molecules_per_minute"] = round(self.stats["total_molecules"] / (elapsed / 60), 2) if elapsed > 0 else 0
        return stats
    
    def print_summary(self):
        """打印运行摘要"""
        stats = self.get_stats()
        summary = f"""
========================================
MCTS运行摘要
========================================
总生成分子数: {stats['total_molecules']}
EGNN迭代轮次: {stats['egnn_iterations']}
最大MCTS深度: {stats['mcts_depth_max']}
随机分子数: {stats['random_molecules']}
运行时间: {stats['elapsed_time']:.2f}秒
生成速率: {stats['molecules_per_minute']:.2f} 分子/分钟
========================================
"""
        print(summary)
        self._write_to_log(summary)
    
    def close(self):
        """关闭日志，刷新缓冲区"""
        self._flush_debug_buffer()
        self.print_summary()
        print(f"[MCTSLogger] 日志保存: {self.log_file}")
        print(f"[MCTSLogger] 调试日志: {self.debug_file}")


# 全局日志实例
_global_logger: Optional[MCTSLogger] = None


def init_logger(target_name: str, log_dir: Optional[Path] = None) -> MCTSLogger:
    """初始化全局日志管理器"""
    global _global_logger
    _global_logger = MCTSLogger(target_name, log_dir)
    return _global_logger


def get_logger() -> Optional[MCTSLogger]:
    """获取全局日志管理器"""
    return _global_logger


def log_progress(egnn_iter: int, mcts_depth: int, random_count: int, 
                 sequence: str = "", energy: float = None):
    """便捷函数：记录进度"""
    if _global_logger:
        _global_logger.log_progress(egnn_iter, mcts_depth, random_count, sequence, energy)


def log_debug(component: str, message: str, data: Optional[Dict] = None):
    """便捷函数：记录调试信息"""
    if _global_logger:
        _global_logger.log_debug(component, message, data)


def log_crosslinker_debug(cys_sulfur_indices: List[int], 
                          selected_sulfur_indices: List[int],
                          br_indices: List[int],
                          bonds_created: int,
                          bond_details: List[Dict] = None):
    """便捷函数：记录交联剂调试信息"""
    if _global_logger:
        _global_logger.log_crosslinker_debug(
            cys_sulfur_indices, selected_sulfur_indices, 
            br_indices, bonds_created, bond_details
        )


def close_logger():
    """便捷函数：关闭日志"""
    global _global_logger
    if _global_logger:
        _global_logger.close()
        _global_logger = None


if __name__ == "__main__":
    # 测试
    logger = init_logger("test_target")
    
    # 模拟100次分子生成
    for i in range(105):
        logger.log_progress(
            egnn_iter=(i // 10) + 1,
            mcts_depth=(i % 10) + 1,
            random_count=i + 1,
            sequence=f"AC{i:03d}C{i:03d}CG",
            energy=-5.5 - (i * 0.01)
        )
        
        # 模拟交联剂调试信息
        logger.log_crosslinker_debug(
            cys_sulfur_indices=[8, 35, 60],
            selected_sulfur_indices=[8, 35, 60],
            br_indices=[72, 76, 79],
            bonds_created=3,
            bond_details=[
                {"s": 8, "c": 71},
                {"s": 35, "c": 74},
                {"s": 60, "c": 74}
            ]
        )
        
        time.sleep(0.01)
    
    close_logger()
