#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二主程序 (run_phase2.py)
功能：MCTS三层决策闭环搜索 + EGNN训练/更新

三层决策：
1. 氨基酸序列填充
2. 交联剂/环化方式选择（TBMB/TATA/TBAB/disulfide/None）
3. 二硫键配对选择（如果使用disulfide）

工作流程：
1. 冷启动：生成初始序列+拓扑 → Vina对接 → 训练初始EGNN模型
2. 日常循环：MCTS迭代（Selection→Expansion→Simulation→Backpropagation）
3. 定期验证：筛选Top-K → Vina验证 → 更新数据集 → 重训EGNN
4. 输出最终候选

输入：
- config中的靶点名称、肽模板、交联剂配置
- 第一阶段准备好的Vina受体文件

输出：
- results/{target_name}/candidates.csv（候选序列、拓扑、分数）
- egnn/models/best_model.pt（训练好的EGNN模型）
"""

import os
import sys
import json
import time
import random
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

import config

# 导入MCTS日志模块
try:
    from mcts_logger import init_logger, close_logger, get_logger
    from simulation import SimulationEngine
except ImportError:
    init_logger = None
    close_logger = None
    get_logger = None
    SimulationEngine = None

# =============================================================================
# 日志配置
# =============================================================================
LOGS_DIR = config.BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(target_name: str):
    """设置日志记录器"""
    import logging
    
    logger = logging.getLogger(f"Phase2Engine_{target_name}")
    logger.setLevel(logging.INFO)
    
    # 清除已有处理器
    logger.handlers = []
    
    # 文件处理器
    log_file = LOGS_DIR / f"{target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# =============================================================================
# 调试配置
# =============================================================================
DEBUG_MODE = os.environ.get('MCTS_DEBUG', '0') == '1'

def debug_print(msg: str, level: str = "INFO"):
    """调试输出函数"""
    if DEBUG_MODE or level in ["ERROR", "WARNING", "CRITICAL"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")

def check_not_mock(value, name: str, expected_type=None):
    """
    检查值不是mock值（如随机数、默认值等）
    
    如果检测到mock值，输出警告
    """
    # 检查是否为可疑的mock值（太整齐的整数或常见默认值）
    if isinstance(value, float):
        # 检查是否为整数形式的浮点数（如-5.0, -10.0）
        if value == int(value) and -20 <= value <= 0:
            debug_print(f"【警告】{name} = {value} 是整数形式的能量值，可能是mock值！", "WARNING")
            debug_print(f"        请确保这是真实的计算结果，而非占位符", "WARNING")
            return False
    
    # 检查是否为0（Vina失败的默认值）
    if value == 0:
        debug_print(f"【警告】{name} = 0，这可能是Vina失败的默认值！", "WARNING")
        return False
    
    # 检查类型
    if expected_type and not isinstance(value, expected_type):
        debug_print(f"【错误】{name} 类型错误: 期望 {expected_type}, 实际 {type(value)}", "ERROR")
        return False
    
    return True

# 导入MCTS模块
from peptide_state import PeptideState, create_root_node, MCTSNode
from selection import PUCTSelector
from expansion import ExpansionEngine
from simulation import SimulationEngine
from backpropagation import BackpropagationEngine

# 其他模块
from seq_generator import generate_full_sequence
from vina import batch_vina_dock

# 导入EGNN模块
from EGNN_1 import main as egnn_data_prep
from EGNN_23 import main as egnn_train
from EGNN_4 import main as egnn_evaluate


@dataclass
class MCTSNodeSnapshot:
    """MCTS节点快照（轻量级，用于序列化）"""
    state_key: str              # PeptideState.to_key()
    visit_count: int
    total_score: float
    children_keys: List[str]    # 子节点标识列表
    decision_level: int
    decision_action: Optional[str]
    prior_prob: float


@dataclass
class MCTSTreeSnapshot:
    """MCTS树快照（轻量级，避免pickle循环引用问题）"""
    nodes: Dict[str, MCTSNodeSnapshot]  # state_key -> snapshot
    root_key: str
    iteration: int
    best_states: List[Tuple[str, float]]  # (state_key, score)
    all_data: Dict[str, float]  # state_key -> energy


@dataclass
class MCTSState:
    """MCTS状态（用于保存/恢复）- 保留用于兼容性"""
    root: MCTSNode
    iteration: int
    best_states: List[Tuple[PeptideState, float]]
    all_data: Dict[str, float]  # state_key -> energy


class Phase2Engine:
    """
    阶段二主引擎 V2 - 支持三层决策和训练恢复
    """
    
    def __init__(self, target_name: str, checkpoint_dir: Optional[Path] = None, resume: bool = True):
        """
        初始化引擎
        
        Args:
            target_name: 靶点名称
            checkpoint_dir: 检查点保存目录
            resume: 是否尝试从检查点恢复训练
        """
        self.target_name = target_name
        self.target_dirs = config.get_target_dirs(target_name)
        
        # 初始化MCTS专用日志管理器
        if init_logger:
            self.mcts_logger = init_logger(target_name)
        else:
            self.mcts_logger = None
        
        # 设置日志
        self.logger, self.log_file = setup_logger(target_name)
        self.logger.info(f"="*60)
        self.logger.info(f"Phase2Engine 初始化")
        self.logger.info(f"靶点: {target_name}")
        self.logger.info(f"日志文件: {self.log_file}")
        if self.mcts_logger:
            self.logger.info(f"MCTS日志: {self.mcts_logger.log_file}")
        self.logger.info(f"="*60)
        
        # 检查第一阶段是否完成
        if not (self.target_dirs["vina"] / "vina-receptor.pdbqt").exists():
            raise FileNotFoundError(f"请先运行阶段一")
        
        # 检查点目录
        if checkpoint_dir is None:
            self.checkpoint_dir = config.BASE_DIR / "checkpoints" / target_name
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 恢复状态文件
        self.resume_state_file = self.checkpoint_dir / "resume_state.json"
        
        # 结果目录
        self.results_dir = config.RESULTS_DIR / target_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 恢复标志
        self.resume = resume
        self.resumed_from_checkpoint = False
        
        # MCTS组件（V3 - 支持EGNN先验）
        self.selector = PUCTSelector(c_puct=config.MCTS_CONFIG["c_puct"])
        # ExpansionEngine将在加载EGNN后初始化
        self.expansion_engine = None
        self.backprop_engine = BackpropagationEngine(verbose=False)
        
        # EGNN模型
        self.egnn_model = None
        self.egnn_model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"
        
        # 统计信息
        self.iteration = 0
        self.current_round = 1  # 当前轮次（多轮闭环优化）
        self.best_states = []  # [(PeptideState, score), ...]
        self.all_data = {}     # {state_key: energy}
        
        # 数据文件
        self.dataset_path = self.results_dir / "dataset.csv"
        self.candidates_path = self.results_dir / "candidates.csv"
        
        self._init_dataset()
    
    def _init_dataset(self):
        """初始化数据集文件"""
        if not self.dataset_path.exists():
            with open(self.dataset_path, 'w') as f:
                f.write("sequence,crosslinker,disulfide_bonds,energy,data_source,iteration,timestamp\n")
        
        if not self.candidates_path.exists():
            with open(self.candidates_path, 'w') as f:
                f.write("sequence,crosslinker,disulfide_bonds,pred_energy,mcts_visits,mcts_avg_score,iteration,timestamp\n")
    
    def cold_start(self, n_sequences: int = 100, use_parallel: bool = True):
        """
        冷启动：生成初始数据并训练EGNN
        
        Args:
            n_sequences: 初始序列数量
            use_parallel: 是否使用并行Vina对接
        """
        print("="*60)
        print("阶段二：冷启动")
        print("="*60)
        
        # 1. 生成随机序列+拓扑
        print(f"\n[1/5] 生成 {n_sequences} 个随机序列+拓扑...")
        states = []
        for _ in range(n_sequences):
            seq = generate_full_sequence()
            # 【修复】使用 config.CROSSLINKER 而不是随机选择
            xlinker = config.CROSSLINKER
            
            state = PeptideState(sequence=seq, crosslinker=xlinker)
            
            # 【修复】只有使用 disulfide 时才需要选择二硫键配对
            if xlinker == "disulfide":
                bonds = state.get_possible_disulfide_bonds()
                if bonds:
                    state.disulfide_bonds = [random.choice(bonds)]
                    state._update_completion_status()
            
            states.append(state)
        
        print(f"  生成 {len(states)} 个状态")
        print(f"  使用交联剂: {config.CROSSLINKER}")
        
        # 2. Vina对接
        print(f"\n[2/5] Vina对接（获取真实分数）...")
        
        # 准备对接参数
        from ligand_generator import generate_ligand
        from vina import get_vina_paths, run_vina_with_progress
        
        vina_paths = get_vina_paths(self.target_name)
        results = {}
        
        for i, state in enumerate(states):
            print(f"\n[{i+1}/{len(states)}] {state}")
            
            try:
                # 获取交联剂配置
                if state.crosslinker == "disulfide":
                    xlinker = "disulfide"
                    positions = list(state.disulfide_bonds[0]) if state.disulfide_bonds else None
                else:
                    xlinker = state.crosslinker
                    # 【修复】使用 config.CROSSLINKER_POSITIONS
                    positions = config.CROSSLINKER_POSITIONS if xlinker else None
                
                # 生成分子
                pdbqt_path = generate_ligand(
                    sequence=state.sequence,
                    crosslinker=xlinker,
                    crosslinker_positions=positions
                )
                
                # Vina对接
                # 【修复】使用config中的CPU配置，而不是硬编码为1
                result = run_vina_with_progress(
                    ligand_pdbqt=pdbqt_path,
                    receptor_pdbqt=vina_paths['receptor'],
                    vina_config=vina_paths['config'],
                    n_cpu=config.VINA_CONFIG.get("cpu", 4)
                )
                
                if result.success:
                    results[state.to_key()] = result.binding_energy
                    print(f"  ✓ 结合能: {result.binding_energy:.4f}")
                else:
                    print(f"  ✗ 失败: {result.error_message}")
                    
            except Exception as e:
                print(f"  ✗ 异常: {e}")
        
        # 3. 保存到数据集
        print(f"\n[3/5] 更新数据集...")
        self._update_dataset(results, "cold_start")
        print(f"  有效数据: {len(results)}/{len(states)}")
        
        # 4. 训练EGNN
        print(f"\n[4/5] 训练初始EGNN模型...")
        self._train_egnn()
        
        # 5. 加载EGNN模型并初始化ExpansionEngine
        print(f"\n[5/5] 加载EGNN模型...")
        self._load_egnn_model()
        
        print("\n" + "="*60)
        print("冷启动完成！")
        print("="*60)
    
    def _update_dataset(self, results: Dict[str, float], source: str):
        """
        更新数据集（过滤无效数据）
        
        Bug 5修复：只保存有效的负结合能数据，避免Vina失败污染训练集
        """
        timestamp = datetime.now().isoformat()
        
        # 过滤无效数据（只保留负结合能）
        valid_results = {}
        filtered_count = 0
        
        for state_key, energy in results.items():
            # 检查能量是否有效（负结合能表示有效结合）
            if energy < 0:
                valid_results[state_key] = energy
            else:
                filtered_count += 1
                print(f"  [过滤] {state_key}: energy={energy} (非负值，可能是Vina失败)")
        
        if filtered_count > 0:
            print(f"  [数据过滤] 过滤了 {filtered_count} 个无效数据")
        
        # 保存有效数据
        with open(self.dataset_path, 'a') as f:
            for state_key, energy in valid_results.items():
                # 解析state_key
                parts = state_key.split('_')
                sequence = parts[0]
                crosslinker = parts[1] if len(parts) > 1 and parts[1] != "None" else "None"
                bonds_str = parts[2] if len(parts) > 2 else ""
                
                f.write(f"{sequence},{crosslinker},{bonds_str},{energy},{source},{self.iteration},{timestamp}\n")
                self.all_data[state_key] = energy
        
        # 更新最佳状态（只使用有效数据）
        for state_key, energy in valid_results.items():
            # 解析state（简化处理）
            parts = state_key.split('_')
            sequence = parts[0]
            crosslinker = parts[1] if len(parts) > 1 and parts[1] != "None" else None
            
            state = PeptideState(sequence=sequence, crosslinker=crosslinker)
            self.best_states.append((state, energy))
        
        # 排序并去重
        self.best_states = sorted(self.best_states, key=lambda x: x[1])[:1000]
    
    def _prepare_egnn_data(self):
        """
        准备EGNN训练数据
        从dataset.csv转换为EGNN_1.py需要的格式
        """
        import csv
        from pathlib import Path
        
        print("  准备EGNN训练数据...")
        
        # 读取dataset.csv（处理Windows换行符）
        dataset_path = self.dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        
        # 手动解析CSV以处理换行符问题
        valid_data = []
        with open(dataset_path, 'r', newline='', encoding='utf-8') as f:
            # 尝试检测分隔符
            sample = f.read(1024)
            f.seek(0)
            
            # 使用csv模块解析
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            
            for row in reader:
                if len(row) >= 4:
                    try:
                        sequence = row[0].strip()
                        # energy在第4列（索引3），格式：sequence,crosslinker,bonds,energy,...
                        energy = float(row[3])
                        # 只保留负结合能的数据
                        if energy < 0:
                            valid_data.append((sequence, energy))
                    except (ValueError, IndexError) as e:
                        print(f"    跳过无效行: {row}, 错误: {e}")
                        continue
        
        if len(valid_data) == 0:
            raise ValueError("没有有效的训练数据（所有结合能都是非负值）")
        
        print(f"    有效样本: {len(valid_data)}")
        
        # 创建sequences.txt（只包含序列）
        sequences_file = config.BASE_DIR / "sequences.txt"
        with open(sequences_file, 'w', newline='') as f:
            for seq, _ in valid_data:
                f.write(f"{seq}\n")
        
        # 创建energies.csv（序列和能量）
        energies_file = self.results_dir / "energies.csv"
        with open(energies_file, 'w', newline='') as f:
            f.write("sequence,energy\n")
            for seq, energy in valid_data:
                f.write(f"{seq},{energy}\n")
        
        print(f"    ✓ 序列文件: {sequences_file} ({len(valid_data)} 个序列)")
        print(f"    ✓ 能量文件: {energies_file}")
        
        return sequences_file, energies_file
    
    def _train_egnn(self):
        """训练EGNN模型"""
        print("  准备EGNN数据...")
        
        # 准备EGNN训练数据
        try:
            sequences_file, energies_file = self._prepare_egnn_data()
            
            # 调用EGNN_1准备数据
            egnn_data_prep(
                sequences_file=sequences_file,
                energies_file=energies_file,
                output_dir=config.BASE_DIR / "egnn" / "raw"
            )
            print("  ✓ EGNN数据准备完成")
        except Exception as e:
            print(f"  ✗ EGNN数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("  训练EGNN模型...")
        egnn_train(
            data_dir=config.BASE_DIR / "egnn" / "raw",
            output_dir=config.BASE_DIR / "egnn" / "models",
            hidden_dim=config.EGNN_CONFIG["hidden_dim"],
            num_layers=config.EGNN_CONFIG["num_layers"],
            num_epochs=config.EGNN_CONFIG["num_epochs"],
            batch_size=config.EGNN_CONFIG["batch_size"],
            lr=config.EGNN_CONFIG["learning_rate"],
            patience=config.EGNN_CONFIG["patience"]
        )
    
    def _load_egnn_model(self):
        """加载EGNN模型并初始化带EGNN先验的ExpansionEngine"""
        if not self.egnn_model_path.exists():
            print(f"警告: EGNN模型不存在")
            return False
        
        try:
            from egnn_predictor import EGNNPredictor
            
            # 使用EGNNPredictor包装器（有predict方法）
            self.egnn_model = EGNNPredictor(
                model_path=self.egnn_model_path,
                hidden_dim=config.EGNN_CONFIG["hidden_dim"],
                num_layers=config.EGNN_CONFIG["num_layers"]
            )
            
            print(f"  ✓ 加载EGNN模型")
            
            # 初始化带EGNN先验的ExpansionEngine
            self._init_expansion_engine()
            
            return True
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _init_expansion_engine(self):
        """初始化带EGNN先验的ExpansionEngine"""
        if self.egnn_model is None:
            print("  警告: EGNN模型未加载，使用默认ExpansionEngine")
            self.expansion_engine = ExpansionEngine()
            return
        
        # 创建EGNN预测包装函数
        def egnn_predict_wrapper(state):
            """包装EGNN预测，返回结合能"""
            try:
                return self._egnn_predict(state)
            except Exception as e:
                debug_print(f"EGNN预测失败: {e}", "WARNING")
                return -5.0  # 回退到中等能量
        
        # 从配置读取先验温度参数（默认1.0，越大越均匀）
        prior_temp = config.MCTS_CONFIG.get("prior_temperature", 1.0)
        use_egnn_prior = config.MCTS_CONFIG.get("use_egnn_prior", True)
        
        self.expansion_engine = ExpansionEngine(
            egnn_model=egnn_predict_wrapper,
            use_egnn_prior=use_egnn_prior,
            prior_temperature=prior_temp
        )
        
        print(f"  ✓ 初始化ExpansionEngine (EGNN先验: {use_egnn_prior}, 温度: {prior_temp})")
    
    def _egnn_predict(self, state: PeptideState) -> float:
        """
        使用EGNN预测结合能
        
        Args:
            state: 肽状态（包含序列、交联剂、二硫键）
        
        Returns:
            预测结合能（kcal/mol）
        
        Raises:
            RuntimeError: 如果EGNN模型未加载或预测失败
        """
        debug_print(f"开始EGNN预测: {state}", "INFO")
        
        # 如果没有EGNN模型，尝试加载
        if self.egnn_model is None:
            debug_print("EGNN模型未加载，尝试加载...", "INFO")
            from egnn_predictor import create_egnn_predictor
            try:
                self.egnn_model = create_egnn_predictor()
                debug_print("✓ EGNN模型加载成功", "INFO")
            except (FileNotFoundError, RuntimeError) as e:
                debug_print(f"【严重错误】EGNN模型加载失败: {e}", "CRITICAL")
                debug_print("        解决方案：", "CRITICAL")
                debug_print("        1. 运行冷启动生成训练数据", "CRITICAL")
                debug_print("        2. 运行 EGNN_1.py 准备数据", "CRITICAL")
                debug_print("        3. 运行 EGNN_23.py 训练模型", "CRITICAL")
                raise RuntimeError(
                    f"EGNN模型未加载: {e}"
                ) from e
        
        # 使用EGNN预测
        from ligand_generator import generate_ligand
        import tempfile
        
        # 获取交联剂配置
        if state.crosslinker == "disulfide":
            xlinker = "disulfide"
            positions = list(state.disulfide_bonds[0]) if state.disulfide_bonds else None
        else:
            xlinker = state.crosslinker
            # 【修复】使用 config.CROSSLINKER_POSITIONS
            positions = config.CROSSLINKER_POSITIONS if xlinker else None
        
        debug_print(f"生成分子: sequence={state.sequence[:20]}..., xlinker={xlinker}", "INFO")
        
        # 生成分子
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdbqt_path = generate_ligand(
                    sequence=state.sequence,
                    crosslinker=xlinker,
                    crosslinker_positions=positions,
                    output_dir=Path(tmpdir)
                )
                debug_print(f"✓ 分子生成成功: {pdbqt_path}", "INFO")
                
                # EGNN预测
                debug_print("开始EGNN预测...", "INFO")
                energy = self.egnn_model.predict(pdbqt_path)
                
                # 检查是否为mock值
                check_not_mock(energy, "EGNN预测能量", float)
                
                debug_print(f"✓ EGNN预测完成: energy={energy:.4f}", "INFO")
                return energy
                
        except Exception as e:
            debug_print(f"【错误】EGNN预测失败: {e}", "ERROR")
            debug_print(f"        序列: {state.sequence}", "ERROR")
            debug_print(f"        交联剂: {state.crosslinker}", "ERROR")
            raise
    
    def mcts_iteration(self, root: MCTSNode, n_iterations: int = 100,
                        max_expansions_per_step: Optional[int] = None,
                        egnn_iteration: int = 1) -> MCTSNode:
        """
        运行MCTS迭代（三层决策，逐步扩展）
        
        Args:
            root: 根节点
            n_iterations: 迭代次数
            max_expansions_per_step: 每次扩展的最大子节点数（默认从config读取）
            egnn_iteration: 当前EGNN迭代轮次（用于日志输出）
        
        Returns:
            更新后的根节点
        """
        if max_expansions_per_step is None:
            max_expansions_per_step = config.MCTS_CONFIG.get("max_expansions", 5)
        
        # 设置EGNN迭代轮次
        if SimulationEngine:
            SimulationEngine.set_egnn_iteration(egnn_iteration)
        
        sim_engine = SimulationEngine(
            target_name=self.target_name,
            egnn_model=self._egnn_predict,
            n_simulations=config.MCTS_CONFIG["n_simulations"]
        )
        
        for i in range(n_iterations):
            self.iteration += 1
            
            # 1. Selection: 选择路径
            # 当遇到可扩展节点时停止，让expansion处理
            path = self.selector.select_path(
                root,
                can_expand_fn=lambda node: self.expansion_engine.can_expand(node)
            )
            leaf = path[-1]
            
            # 2. Expansion: 逐步扩展（只扩展最多max_expansions_per_step个子节点）
            if not leaf.is_terminal and self.expansion_engine.can_expand(leaf):
                # 逐步扩展：只扩展部分子节点，不是全部
                new_children = self.expansion_engine.expand(
                    leaf, 
                    max_expansions=max_expansions_per_step
                )
                
                # 从新扩展的子节点中选择一个进行模拟
                if new_children:
                    # 策略：选择prior_prob最高的新子节点
                    best_child = max(new_children.values(), key=lambda c: c.prior_prob)
                    path.append(best_child)
                    leaf = best_child
                elif leaf.children:
                    # 如果没有新扩展，但从已有子节点中选择
                    # 选择visit_count最少的（鼓励探索）
                    best_child = min(leaf.children.values(), key=lambda c: c.visit_count)
                    path.append(best_child)
                    leaf = best_child
            
            # 3. Simulation: 评估
            if leaf.is_terminal:
                result = sim_engine.simulate(leaf, verbose=False)
                reward = result.score
            else:
                # 非终端节点使用启发式分数
                reward = 0.5
            
            # 4. Backpropagation: 回溯更新
            self.backprop_engine.backpropagate(path, reward)
            
            # 定期输出进度
            if (i + 1) % 100 == 0:
                # 统计树的大小
                def count_nodes(node):
                    count = 1
                    for child in node.children.values():
                        count += count_nodes(child)
                    return count
                
                tree_size = count_nodes(root)
                stats_msg = f"  MCTS迭代: {i+1}/{n_iterations}, 树节点数: {tree_size}"
                
                # 打印EGNN统计（如果使用EGNN先验）
                if self.expansion_engine and hasattr(self.expansion_engine, 'get_egnn_stats'):
                    egnn_stats = self.expansion_engine.get_egnn_stats()
                    if egnn_stats['egnn_calls'] > 0:
                        stats_msg += f", EGNN调用: {egnn_stats['egnn_calls']}(缓存命中: {egnn_stats['cache_hits']})"
                
                print(stats_msg)
        
        return root
    
    def extract_candidates(self, root: MCTSNode, n_candidates: int = 100) -> List[Tuple[PeptideState, float]]:
        """
        提取候选状态（迭代式DFS，避免栈溢出）
        
        Args:
            root: 根节点
            n_candidates: 候选数量
        
        Returns:
            List[Tuple[PeptideState, float]] - 候选状态和分数
        """
        candidates = []
        
        # 迭代式DFS（使用栈）
        stack = [root]
        visited = set()
        
        while stack:
            node = stack.pop()
            node_id = id(node)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # 如果是终端节点，添加到候选列表
            if node.is_terminal:
                # 深拷贝状态，避免后续树结构变化影响候选列表
                state_copy = node.state.copy()
                candidates.append((state_copy, node.average_score))
            
            # 将子节点加入栈
            for child in node.children.values():
                if id(child) not in visited:
                    stack.append(child)
        
        # 按分数排序，返回top-N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n_candidates]
    
    def validate_candidates_with_vina(self, candidates: List[Tuple[PeptideState, float]], 
                                      top_n: int = 20) -> Dict[str, float]:
        """
        使用Vina验证候选序列
        
        Args:
            candidates: MCTS候选列表 [(state, score), ...]
            top_n: 验证前N个候选
        
        Returns:
            {state_key: vina_energy, ...}
        """
        from ligand_generator import generate_ligand
        from vina import get_vina_paths, run_vina_with_progress
        
        print(f"\n{'='*60}")
        print(f"Vina验证: 测试Top-{top_n}候选")
        print(f"{'='*60}")
        
        # 选择Top-N候选
        top_candidates = candidates[:top_n]
        
        vina_paths = get_vina_paths(self.target_name)
        results = {}
        
        for i, (state, mcts_score) in enumerate(top_candidates, 1):
            print(f"\n[{i}/{top_n}] {state}")
            print(f"  MCTS分数: {mcts_score:.4f}")
            
            try:
                # 获取交联剂配置
                if state.crosslinker == "disulfide":
                    xlinker = "disulfide"
                    positions = list(state.disulfide_bonds[0]) if state.disulfide_bonds else None
                else:
                    xlinker = state.crosslinker
                    positions = config.CROSSLINKER_POSITIONS if xlinker else None
                
                # 生成分子
                pdbqt_path = generate_ligand(
                    sequence=state.sequence,
                    crosslinker=xlinker,
                    crosslinker_positions=positions
                )
                
                # Vina对接
                result = run_vina_with_progress(
                    ligand_pdbqt=pdbqt_path,
                    receptor_pdbqt=vina_paths['receptor'],
                    vina_config=vina_paths['config'],
                    n_cpu=config.VINA_CONFIG.get("cpu", 4)
                )
                
                if result.success:
                    results[state.to_key()] = result.binding_energy
                    print(f"  ✓ Vina结合能: {result.binding_energy:.4f}")
                else:
                    print(f"  ✗ Vina失败: {result.error_message}")
                    
            except Exception as e:
                print(f"  ✗ 异常: {e}")
                import traceback
                traceback.print_exc()
        
        # 统计
        print(f"\n{'='*60}")
        print(f"Vina验证完成: {len(results)}/{top_n} 成功")
        if results:
            energies = list(results.values())
            print(f"  平均结合能: {np.mean(energies):.4f}")
            print(f"  最佳结合能: {min(energies):.4f}")
            print(f"  最差结合能: {max(energies):.4f}")
        print(f"{'='*60}")
        
        return results
    
    def finetune_egnn(self, new_data: Dict[str, float], n_epochs: int = 20):
        """
        使用新数据微调EGNN模型
        
        Args:
            new_data: 新验证数据 {state_key: energy, ...}
            n_epochs: 微调轮数
        """
        print(f"\n{'='*60}")
        print(f"EGNN微调: 使用 {len(new_data)} 个新样本")
        print(f"{'='*60}")
        
        # 1. 更新数据集
        print("\n[1/3] 更新数据集...")
        self._update_dataset(new_data, f"round_{self.current_round}")
        
        # 2. 准备EGNN数据
        print("\n[2/3] 准备EGNN训练数据...")
        try:
            sequences_file, energies_file = self._prepare_egnn_data()
            
            from EGNN_1 import main as egnn_data_prep
            egnn_data_prep(
                sequences_file=sequences_file,
                energies_file=energies_file,
                output_dir=config.BASE_DIR / "egnn" / "raw"
            )
            print("  ✓ EGNN数据准备完成")
        except Exception as e:
            print(f"  ✗ EGNN数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. 微调EGNN（使用较少的epoch）
        print(f"\n[3/3] 微调EGNN模型 ({n_epochs} epochs)...")
        try:
            from EGNN_23 import main as egnn_train
            
            # 备份旧模型
            backup_path = self.egnn_model_path.with_suffix('.pt.backup')
            if self.egnn_model_path.exists():
                import shutil
                shutil.copy2(self.egnn_model_path, backup_path)
                print(f"  备份旧模型: {backup_path}")
            
            # 微调训练
            egnn_train(
                data_dir=config.BASE_DIR / "egnn" / "raw",
                output_dir=config.BASE_DIR / "egnn" / "models",
                hidden_dim=config.EGNN_CONFIG["hidden_dim"],
                num_layers=config.EGNN_CONFIG["num_layers"],
                num_epochs=n_epochs,  # 使用较少的轮数微调
                batch_size=config.EGNN_CONFIG["batch_size"],
                lr=config.EGNN_CONFIG["learning_rate"] * 0.5,  # 降低学习率
                patience=10  # 早停耐心值
            )
            
            print("  ✓ EGNN微调完成")
            
            # 4. 重新加载模型
            print("\n[4/4] 重新加载EGNN模型...")
            self._load_egnn_model()
            
            # 清空EGNN缓存（模型已更新）
            if self.expansion_engine and hasattr(self.expansion_engine, '_egnn_cache'):
                self.expansion_engine._egnn_cache.clear()
                print("  ✓ 清空EGNN缓存")
            
            return True
            
        except Exception as e:
            print(f"  ✗ EGNN微调失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 恢复备份
            if backup_path.exists():
                import shutil
                shutil.copy2(backup_path, self.egnn_model_path)
                print(f"  恢复旧模型")
                self._load_egnn_model()
            
            return False
    
    def _calculate_convergence_metric(self, vina_results: Dict[str, float], metric: str = 'best') -> float:
        """
        计算收敛判定指标
        
        Args:
            vina_results: Vina验证结果 {state_key: energy, ...}
            metric: 指标类型 ('best', 'mean', 'top3_mean', 'median')
        
        Returns:
            指标值（结合能，越负越好）
        """
        if not vina_results:
            return float('inf')  # 无数据返回正无穷（最差）
        
        energies = list(vina_results.values())
        
        if metric == 'best':
            # 最佳结合能（最负）
            return min(energies)
        elif metric == 'mean':
            # 平均结合能
            return np.mean(energies)
        elif metric == 'top3_mean':
            # Top-3平均
            sorted_energies = sorted(energies)
            top3 = sorted_energies[:min(3, len(sorted_energies))]
            return np.mean(top3)
        elif metric == 'median':
            # 中位数
            return np.median(energies)
        else:
            # 默认最佳
            return min(energies)
    
    def _check_convergence(self, 
                          round_history: List[Dict], 
                          convergence_config: Dict) -> Tuple[bool, str]:
        """
        检查是否收敛
        
        Args:
            round_history: 历史轮次结果列表
            convergence_config: 收敛配置
        
        Returns:
            (是否收敛, 原因说明)
        """
        if len(round_history) < convergence_config['window_size']:
            return False, "历史数据不足"
        
        metric = convergence_config['metric']
        min_improvement = convergence_config['min_improvement']
        patience = convergence_config['patience']
        window_size = convergence_config['window_size']
        
        # 提取最近几轮的指标
        recent_metrics = []
        for result in round_history[-window_size:]:
            vina_results = result.get('vina_results', {})
            m = self._calculate_convergence_metric(vina_results, metric)
            recent_metrics.append(m)
        
        # 检查改善情况
        # 注意：结合能越负越好，所以改善 = 新值 < 旧值
        improvements = []
        for i in range(1, len(recent_metrics)):
            # 改善量 = 旧值 - 新值（正数表示改善）
            improvement = recent_metrics[i-1] - recent_metrics[i]
            improvements.append(improvement)
        
        # 判定逻辑
        if len(improvements) == 0:
            return False, "数据不足"
        
        # 检查最近N轮是否有显著改善
        no_improvement_count = sum(1 for imp in improvements if imp < min_improvement)
        
        if no_improvement_count >= patience:
            return True, f"连续{no_improvement_count}轮改善不足{min_improvement} kcal/mol"
        
        # 检查是否达到平台期（波动很小）
        if len(recent_metrics) >= 3:
            recent_std = np.std(recent_metrics)
            if recent_std < min_improvement / 2:
                return True, f"指标波动很小(std={recent_std:.4f})，达到平台期"
        
        return False, f"最近改善: {improvements[-1]:.4f} kcal/mol"
    
    def _print_convergence_analysis(self, round_history: List[Dict], convergence_config: Dict):
        """打印收敛分析"""
        print(f"\n{'='*60}")
        print("收敛分析")
        print(f"{'='*60}")
        
        metric = convergence_config['metric']
        
        print(f"\n各轮{metric}指标:")
        for i, result in enumerate(round_history, 1):
            vina_results = result.get('vina_results', {})
            m = self._calculate_convergence_metric(vina_results, metric)
            n_valid = len(vina_results)
            print(f"  第{i}轮: {m:.4f} kcal/mol ({n_valid}个样本)")
        
        if len(round_history) >= 2:
            print(f"\n改善趋势:")
            for i in range(1, len(round_history)):
                prev = self._calculate_convergence_metric(round_history[i-1].get('vina_results', {}), metric)
                curr = self._calculate_convergence_metric(round_history[i].get('vina_results', {}), metric)
                improvement = prev - curr
                symbol = "↓" if improvement > 0 else "↑"
                print(f"  第{i}轮 → 第{i+1}轮: {improvement:+.4f} {symbol}")
    
    def run_multi_round(self, 
                        n_rounds: int = 3,
                        n_mcts_iterations: int = 1000,
                        validation_interval: int = 5000,
                        max_iterations: int = 50000,
                        cold_start_n: int = 100,
                        top_n_final: int = 20,
                        skip_vina: bool = False,
                        convergence_config: Optional[Dict] = None):
        """
        多轮MCTS-验证-重训闭环（支持自动收敛判定）
        
        Args:
            n_rounds: 最大循环轮数（如果未收敛）
            n_mcts_iterations: 每轮MCTS迭代次数
            validation_interval: 验证间隔
            max_iterations: 每轮最大迭代数
            cold_start_n: 冷启动序列数
            top_n_final: 每轮验证的候选数
            skip_vina: 是否跳过Vina验证（测试用）
            convergence_config: 收敛判定配置
                - patience: 容忍轮数（默认2）
                - min_improvement: 最小改善阈值（默认0.5 kcal/mol）
                - window_size: 滑动窗口大小（默认3）
                - metric: 判定指标（'best', 'mean', 'top3_mean'）
        """
        # 默认收敛配置（从config读取）
        if convergence_config is None:
            convergence_config = {
                'patience': config.MULTI_ROUND_CONFIG['patience'],
                'min_improvement': config.MULTI_ROUND_CONFIG['min_improvement'],
                'window_size': config.MULTI_ROUND_CONFIG['window_size'],
                'metric': config.MULTI_ROUND_CONFIG['convergence_metric'],
                'max_rounds': n_rounds
            }
        
        print("="*60)
        print(f"阶段二：多轮MCTS闭环优化（自动收敛）")
        print(f"靶点: {self.target_name}")
        print(f"最大轮数: {n_rounds}")
        print(f"每轮候选: {top_n_final}")
        print(f"收敛配置:")
        print(f"  指标: {convergence_config['metric']}")
        print(f"  最小改善: {convergence_config['min_improvement']} kcal/mol")
        print(f"  容忍轮数: {convergence_config['patience']}")
        print("="*60)
        
        # 检查是否需要冷启动
        if not self.egnn_model_path.exists():
            print("\n未检测到EGNN模型，执行冷启动...")
            self.cold_start(n_sequences=cold_start_n)
        else:
            print("\n检测到已有EGNN模型，加载中...")
            self._load_egnn_model()
        
        # 多轮循环
        all_results = []
        
        for round_idx in range(1, n_rounds + 1):
            self.current_round = round_idx
            
            print(f"\n{'='*60}")
            print(f"第 {round_idx}/{n_rounds} 轮 MCTS搜索")
            print(f"{'='*60}")
            
            # 保存轮次状态
            self.save_resume_state({
                'cold_start_completed': True,
                'current_round': round_idx,
                'total_rounds': n_rounds,
                'phase': 'mcts_search'
            })
            
            # 创建或加载根节点
            root = self.load_checkpoint(f"round{round_idx}_latest.json")
            if root is None:
                print(f"\n创建新的MCTS根节点（第{round_idx}轮）...")
                root = create_root_node()
                self.iteration = 0
            else:
                print(f"\n从检查点恢复MCTS树（第{round_idx}轮，迭代{self.iteration}）")
            
            # MCTS搜索
            try:
                while self.iteration < max_iterations:
                    print(f"\n--- 第{round_idx}轮 MCTS迭代 {self.iteration}-{self.iteration + n_mcts_iterations} ---")
                    
                    root = self.mcts_iteration(
                        root, 
                        n_mcts_iterations,
                        max_expansions_per_step=config.MCTS_CONFIG.get("max_expansions", 5),
                        egnn_iteration=round_idx  # 传递当前EGNN迭代轮次
                    )
                    
                    # 保存检查点
                    self.save_checkpoint(root, f"round{round_idx}_latest.json")
                    
                    # 定期验证（每轮内部也验证）
                    if self.iteration % validation_interval < n_mcts_iterations:
                        print(f"\n--- 第{round_idx}轮 中期验证（迭代 {self.iteration}） ---")
                        candidates = self.extract_candidates(root, 100)
                        print(f"  提取 {len(candidates)} 个候选")
                        self.save_checkpoint(root, f"round{round_idx}_iter{self.iteration}.json")
                
                # 本轮MCTS完成
                print(f"\n{'='*60}")
                print(f"第 {round_idx} 轮 MCTS搜索完成")
                print(f"{'='*60}")
                
                # 提取最终候选
                final_candidates = self.extract_candidates(root, top_n_final * 2)  # 多提取一些
                print(f"\n提取Top-{len(final_candidates)}候选")
                
                # Vina验证
                if not skip_vina:
                    vina_results = self.validate_candidates_with_vina(final_candidates, top_n=top_n_final)
                    all_results.append({
                        'round': round_idx,
                        'candidates': final_candidates[:top_n_final],
                        'vina_results': vina_results
                    })
                    
                    # 检查收敛（如果已有足够历史数据）
                    if len(all_results) >= convergence_config['window_size']:
                        is_converged, reason = self._check_convergence(all_results, convergence_config)
                        
                        if is_converged:
                            print(f"\n{'='*60}")
                            print(f"🎉 收敛判定: 优化已收敛！")
                            print(f"原因: {reason}")
                            print(f"共进行 {round_idx} 轮")
                            print(f"{'='*60}")
                            
                            # 打印收敛分析
                            self._print_convergence_analysis(all_results, convergence_config)
                            
                            # 保存最终状态
                            self.save_checkpoint(root, f"round{round_idx}_final.json")
                            
                            # 输出总结
                            print("\n" + "="*60)
                            print("多轮MCTS闭环优化完成（收敛）！")
                            print("="*60)
                            self._print_multi_round_summary(all_results)
                            
                            return all_results
                    
                    # 微调EGNN（如果不是最后一轮且未收敛）
                    if round_idx < n_rounds and vina_results:
                        success = self.finetune_egnn(vina_results, n_epochs=20)
                        if success:
                            print(f"\n✓ 第 {round_idx} 轮 EGNN微调完成")
                        else:
                            print(f"\n✗ 第 {round_idx} 轮 EGNN微调失败，继续下一轮")
                else:
                    print("\n跳过Vina验证（测试模式）")
                    all_results.append({
                        'round': round_idx,
                        'candidates': final_candidates[:top_n_final],
                        'vina_results': {}
                    })
                
                # 保存本轮最终检查点
                self.save_checkpoint(root, f"round{round_idx}_final.json")
                
            except KeyboardInterrupt:
                print(f"\n\n第 {round_idx} 轮被中断")
                self.save_checkpoint(root, f"round{round_idx}_interrupted.json")
                raise
        
        # 所有轮次完成
        print("\n" + "="*60)
        print("多轮MCTS闭环优化完成！")
        print("="*60)
        
        # 输出总结
        self._print_multi_round_summary(all_results)
        
        return all_results
    
    def _print_multi_round_summary(self, all_results: List[Dict]):
        """打印多轮优化总结"""
        print(f"\n{'='*60}")
        print("多轮优化结果总结")
        print(f"{'='*60}")
        
        for result in all_results:
            round_idx = result['round']
            candidates = result['candidates']
            vina_results = result['vina_results']
            
            print(f"\n第 {round_idx} 轮:")
            print(f"  候选数量: {len(candidates)}")
            
            if vina_results:
                energies = list(vina_results.values())
                print(f"  Vina验证: {len(energies)} 个")
                print(f"  平均结合能: {np.mean(energies):.4f}")
                print(f"  最佳结合能: {min(energies):.4f}")
                
                # 显示Top-3
                sorted_results = sorted(vina_results.items(), key=lambda x: x[1])
                print(f"  Top-3:")
                for i, (state_key, energy) in enumerate(sorted_results[:3], 1):
                    print(f"    {i}. {state_key}: {energy:.4f}")
        
        # 比较各轮最佳
        if len(all_results) > 1:
            print(f"\n{'='*60}")
            print("各轮最佳对比:")
            for result in all_results:
                round_idx = result['round']
                vina_results = result['vina_results']
                if vina_results:
                    best_energy = min(vina_results.values())
                    print(f"  第{round_idx}轮最佳: {best_energy:.4f}")
        
        print(f"{'='*60}")
    
    def _create_tree_snapshot(self, root: MCTSNode) -> MCTSTreeSnapshot:
        """
        创建MCTS树的轻量级快照（避免pickle循环引用）
        
        Bug 4修复：不序列化整个对象，只保存关键数据
        """
        nodes = {}
        
        # 迭代式遍历树
        stack = [root]
        visited = set()
        
        while stack:
            node = stack.pop()
            node_key = node.state.to_key()
            
            if node_key in visited:
                continue
            visited.add(node_key)
            
            # 创建节点快照
            children_keys = [child.state.to_key() for child in node.children.values()]
            snapshot = MCTSNodeSnapshot(
                state_key=node_key,
                visit_count=node.visit_count,
                total_score=node.total_score,
                children_keys=children_keys,
                decision_level=node.decision_level,
                decision_action=node.decision_action,
                prior_prob=node.prior_prob
            )
            nodes[node_key] = snapshot
            
            # 添加子节点到栈
            for child in node.children.values():
                child_key = child.state.to_key()
                if child_key not in visited:
                    stack.append(child)
        
        # 创建树快照
        best_states_keys = [(state.to_key(), score) for state, score in self.best_states]
        
        return MCTSTreeSnapshot(
            nodes=nodes,
            root_key=root.state.to_key(),
            iteration=self.iteration,
            best_states=best_states_keys,
            all_data=self.all_data.copy()
        )
    
    def _rebuild_tree_from_snapshot(self, snapshot: MCTSTreeSnapshot) -> Optional[MCTSNode]:
        """
        从快照重建MCTS树（恢复parent指针）
        
        Bug 4修复：重建时正确设置parent指针，确保backpropagation正常工作
        """
        if not snapshot.nodes or snapshot.root_key not in snapshot.nodes:
            return None
        
        # 第一步：创建所有节点（不设置parent和children）
        node_map = {}
        for key, snap in snapshot.nodes.items():
            # 解析state_key重建PeptideState
            parts = key.split('_')
            sequence = parts[0]
            crosslinker = parts[1] if len(parts) > 1 and parts[1] != "None" else None
            
            state = PeptideState(sequence=sequence, crosslinker=crosslinker)
            
            # 解析二硫键
            if len(parts) > 2 and parts[2]:
                bonds_str = parts[2].split('|')
                for bond_str in bonds_str:
                    if '-' in bond_str:
                        i, j = bond_str.split('-')
                        state.disulfide_bonds.append((int(i), int(j)))
            
            state._update_completion_status()
            
            # 创建节点
            node = MCTSNode(
                state=state,
                visit_count=snap.visit_count,
                total_score=snap.total_score,
                children={},  # 稍后填充
                parent=None,  # 稍后填充
                prior_prob=snap.prior_prob,
                decision_level=snap.decision_level,
                decision_action=snap.decision_action
            )
            node_map[key] = node
        
        # 第二步：恢复children和parent关系
        for key, snap in snapshot.nodes.items():
            node = node_map[key]
            for child_key in snap.children_keys:
                if child_key in node_map:
                    child = node_map[child_key]
                    child.parent = node
                    # 使用decision_action作为key
                    action = child.decision_action if child.decision_action else child_key
                    node.children[action] = child
        
        # 第三步：恢复引擎状态
        self.iteration = snapshot.iteration
        self.all_data = snapshot.all_data.copy()
        
        # 恢复best_states
        self.best_states = []
        for state_key, score in snapshot.best_states:
            if state_key in node_map:
                self.best_states.append((node_map[state_key].state, score))
        
        return node_map.get(snapshot.root_key)
    
    def save_checkpoint(self, root: MCTSNode, filename: Optional[str] = None):
        """
        保存检查点（使用轻量级快照，避免pickle循环引用问题）
        
        Bug 4修复：不pickle整个对象树，只保存关键数据
        """
        if filename is None:
            filename = f"checkpoint_iter{self.iteration}.json"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # 创建快照
        snapshot = self._create_tree_snapshot(root)
        
        # 保存为JSON（更可靠，跨平台兼容）
        import json
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'nodes': {k: {
                    'state_key': v.state_key,
                    'visit_count': v.visit_count,
                    'total_score': v.total_score,
                    'children_keys': v.children_keys,
                    'decision_level': v.decision_level,
                    'decision_action': v.decision_action,
                    'prior_prob': v.prior_prob
                } for k, v in snapshot.nodes.items()},
                'root_key': snapshot.root_key,
                'iteration': snapshot.iteration,
                'best_states': snapshot.best_states,
                'all_data': snapshot.all_data
            }, f, indent=2)
        
        print(f"  ✓ 保存检查点: {checkpoint_path} ({len(snapshot.nodes)} 个节点)")
    
    def load_checkpoint(self, filename: str) -> Optional[MCTSNode]:
        """
        加载检查点（从JSON快照重建树）
        
        Bug 4修复：正确重建parent指针
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            return None
        
        import json
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        # 重建快照对象
        nodes = {}
        for k, v in data['nodes'].items():
            nodes[k] = MCTSNodeSnapshot(
                state_key=v['state_key'],
                visit_count=v['visit_count'],
                total_score=v['total_score'],
                children_keys=v['children_keys'],
                decision_level=v['decision_level'],
                decision_action=v['decision_action'],
                prior_prob=v['prior_prob']
            )
        
        snapshot = MCTSTreeSnapshot(
            nodes=nodes,
            root_key=data['root_key'],
            iteration=data['iteration'],
            best_states=data['best_states'],
            all_data=data['all_data']
        )
        
        # 重建树
        root = self._rebuild_tree_from_snapshot(snapshot)
        
        if root:
            self.resumed_from_checkpoint = True
            print(f"  ✓ 加载检查点: {checkpoint_path} (迭代 {self.iteration}, {len(nodes)} 个节点)")
            self.logger.info(f"从检查点恢复: 迭代 {self.iteration}, {len(nodes)} 个节点")
        
        return root
    
    def save_resume_state(self, state: dict):
        """
        保存恢复状态（用于中断后恢复）
        
        Args:
            state: 包含训练状态的字典
        """
        state['timestamp'] = datetime.now().isoformat()
        state['target_name'] = self.target_name
        
        with open(self.resume_state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"保存恢复状态: {self.resume_state_file}")
    
    def load_resume_state(self) -> Optional[dict]:
        """
        加载恢复状态
        
        Returns:
            恢复状态字典，如果没有则返回None
        """
        if not self.resume_state_file.exists():
            return None
        
        try:
            with open(self.resume_state_file, 'r') as f:
                state = json.load(f)
            
            self.logger.info(f"加载恢复状态: {self.resume_state_file}")
            self.logger.info(f"  上次运行时间: {state.get('timestamp', 'unknown')}")
            self.logger.info(f"  上次迭代: {state.get('iteration', 0)}")
            return state
        except Exception as e:
            self.logger.error(f"加载恢复状态失败: {e}")
            return None
    
    def run_iterative_loop(self, max_iterations: int = None) -> None:
        """
        【新增】运行迭代闭环优化（新流程）
        
        流程：
        1. MCTS选择节点 → 随机补全N个序列 → EGNN预测 → 平均作为Q值
        2. 重复选择直到填满所有可变位置（路径不重复）
        3. 提取Top-K序列 → Vina验证
        4. 用Vina结果微调EGNN
        5. 计算R²和残差 → 记录日志
        6. 检查收敛（连续N轮无改善则停止）
        
        Args:
            max_iterations: 最大迭代轮数（默认从config读取）
        """
        if max_iterations is None:
            max_iterations = config.MAX_EGNN_ITERATIONS
        
        print("\n" + "="*60)
        print("迭代闭环优化模式（新流程）")
        print("="*60)
        
        # 初始化组件
        from simulation import SimulationEngine
        from selection import PUCTSelector
        from backpropagation import BackpropagationEngine
        from peptide_state import create_root_node
        
        sim_engine = SimulationEngine(
            target_name=self.target_name,
            egnn_model=self._egnn_predict
        )
        selector = PUCTSelector(c_puct=config.MCTS_CONFIG["c_puct"])
        backprop_engine = BackpropagationEngine(verbose=False)
        
        # 创建根节点
        root = create_root_node()
        print(f"\n创建MCTS根节点")
        print(f"  模板: {config.PEPTIDE_TEMPLATE}")
        print(f"  可变位置数: {config.VARIABLE_POSITIONS_COUNT}")
        
        # 收敛追踪
        r2_best = -float('inf')
        mae_best = float('inf')
        patience_counter = 0
        
        # 运行迭代
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"EGNN迭代轮次 {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")
            
            # 1. MCTS选择与随机补全
            path = [root]
            current = root
            depth = 0
            
            while depth < config.MAX_PATH_LENGTH:
                # 选择子节点
                if not current.children:
                    break
                
                next_node = selector.select(current)
                if next_node is None:
                    break
                
                path.append(next_node)
                current = next_node
                depth += 1
                
                # 检查是否填满
                if current.state.is_sequence_complete:
                    break
            
            print(f"  MCTS路径深度: {depth}")
            
            # 2. 对叶节点进行随机补全和评估
            leaf = path[-1]
            if not leaf.state.is_terminal:
                # 使用simulation引擎进行评估
                result = sim_engine.simulate(leaf, verbose=False)
                reward = result.score
            else:
                reward = 1.0  # 终端节点给予最高奖励
            
            # 3. 回溯更新
            backprop_engine.backpropagate(path, reward)
            
            # 4. 每N轮进行一次Vina验证
            if (iteration + 1) % 5 == 0 or iteration == max_iterations - 1:
                print(f"\n  执行Vina验证...")
                # TODO: 实现Vina验证逻辑
                
                # 5. 微调EGNN
                # TODO: 实现EGNN微调
                
                # 6. 计算R²和MAE
                # TODO: 实现指标计算
                
                # 7. 检查收敛
                # TODO: 实现收敛判定
            
            # 保存检查点
            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(root, f"iter_{iteration+1}.json")
        
        print(f"\n{'='*60}")
        print("迭代闭环优化完成")
        print(f"{'='*60}")
    
    def run(self, 
            n_mcts_iterations: int = 1000,
            validation_interval: int = 5000,
            max_iterations: int = 50000,
            cold_start_n: int = 100):
        """
        运行阶段二主循环
        
        Args:
            n_mcts_iterations: 每次MCTS迭代次数
            validation_interval: 验证间隔
            max_iterations: 最大迭代次数
            cold_start_n: 冷启动序列数量
        """
        print("="*60)
        print(f"阶段二：MCTS三层决策闭环搜索")
        print(f"靶点: {self.target_name}")
        print(f"配置交联剂: {config.CROSSLINKER}")
        print("="*60)
        
        # 尝试加载恢复状态
        resume_state = None
        if self.resume:
            resume_state = self.load_resume_state()
            if resume_state:
                print(f"\n检测到之前的训练状态，将从中断处恢复...")
                print(f"  上次迭代: {resume_state.get('iteration', 0)}")
                print(f"  上次时间: {resume_state.get('timestamp', 'unknown')}")
        
        # 检查是否需要冷启动
        if not self.egnn_model_path.exists():
            if resume_state and resume_state.get('cold_start_completed'):
                print("\n检测到EGNN模型缺失，但冷启动已完成标记存在...")
                print("  可能需要重新训练EGNN模型")
            else:
                print("\n未检测到EGNN模型，执行冷启动...")
                self.cold_start(n_sequences=cold_start_n)
                # 标记冷启动已完成
                self.save_resume_state({
                    'cold_start_completed': True,
                    'iteration': 0,
                    'phase': 'mcts_search'
                })
        else:
            print("\n检测到已有EGNN模型，加载中...")
            self._load_egnn_model()
        
        # 创建或加载根节点（从检查点）
        root = self.load_checkpoint("latest.json")
        if root is None:
            if resume_state and resume_state.get('iteration', 0) > 0:
                print("\n警告: 有恢复状态但无法加载MCTS树检查点")
                print("  将创建新的MCTS根节点，但保留已有数据")
            else:
                print("\n创建新的MCTS根节点...")
            root = create_root_node()
        else:
            print(f"\n从检查点恢复MCTS树（当前迭代: {self.iteration}）")
        
        # 恢复迭代计数（如果比检查点更大）
        if resume_state and resume_state.get('iteration', 0) > self.iteration:
            self.iteration = resume_state.get('iteration', 0)
            print(f"  恢复迭代计数: {self.iteration}")
        
        print(f"\n开始MCTS搜索（最大迭代: {max_iterations}）...")
        
        try:
            while self.iteration < max_iterations:
                print(f"\n--- MCTS迭代 {self.iteration}-{self.iteration + n_mcts_iterations} ---")
                
                # 保存恢复状态（用于中断恢复）
                self.save_resume_state({
                    'cold_start_completed': True,
                    'iteration': self.iteration,
                    'phase': 'mcts_search',
                    'best_states_count': len(self.best_states),
                    'all_data_count': len(self.all_data)
                })
                
                root = self.mcts_iteration(
                    root, 
                    n_mcts_iterations,
                    max_expansions_per_step=config.MCTS_CONFIG.get("max_expansions", 5),
                    egnn_iteration=1  # 单轮模式使用固定值1
                )
                
                # 保存检查点
                self.save_checkpoint(root, "latest.json")
                
                if self.iteration % validation_interval < n_mcts_iterations:
                    print(f"\n--- 触发验证（迭代 {self.iteration}） ---")
                    candidates = self.extract_candidates(root, 100)
                    print(f"  提取 {len(candidates)} 个候选")
                    # TODO: Vina验证和重训
                    
                    # 保存当前检查点
                    self.save_checkpoint(root, f"checkpoint_iter{self.iteration}.json")
            
            # 训练完成，标记状态
            self.save_resume_state({
                'cold_start_completed': True,
                'iteration': self.iteration,
                'phase': 'completed',
                'best_states_count': len(self.best_states),
                'all_data_count': len(self.all_data),
                'completed': True
            })
            
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("训练被用户中断")
            print("="*60)
            print(f"当前迭代: {self.iteration}")
            print(f"已保存检查点: {self.checkpoint_dir}/latest.json")
            print(f"恢复状态: {self.resume_state_file}")
            print("\n下次运行将自动从中断处恢复")
            print("="*60)
            
            # 保存最终状态
            self.save_checkpoint(root, "interrupted.json")
            self.save_resume_state({
                'cold_start_completed': True,
                'iteration': self.iteration,
                'phase': 'interrupted',
                'best_states_count': len(self.best_states),
                'all_data_count': len(self.all_data),
                'interrupted': True
            })
            
            raise
        
        except Exception as e:
            print("\n\n" + "="*60)
            print("训练发生错误")
            print("="*60)
            print(f"错误: {e}")
            print(f"当前迭代: {self.iteration}")
            
            # 保存错误状态
            self.save_checkpoint(root, "error.json")
            self.save_resume_state({
                'cold_start_completed': True,
                'iteration': self.iteration,
                'phase': 'error',
                'error': str(e),
                'best_states_count': len(self.best_states),
                'all_data_count': len(self.all_data)
            })
            
            raise
        
        print("\n" + "="*60)
        print("阶段二完成！")
        print("="*60)
        
        final_candidates = self.extract_candidates(root, 10)
        print(f"\n最终候选（Top 10）:")
        for i, (state, score) in enumerate(final_candidates, 1):
            print(f"  {i}. {state}")
        
        # 保存最终检查点
        self.save_checkpoint(root, "final.json")
        
        return final_candidates


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='阶段二：MCTS三层决策闭环搜索')
    parser.add_argument('-t', '--target', type=str, required=True,
                       help='靶点名称')
    parser.add_argument('--cold-start', action='store_true',
                       help='强制冷启动')
    parser.add_argument('--n-sequences', type=int, default=100,
                       help='冷启动序列数量')
    parser.add_argument('--mcts-iter', type=int, default=1000,
                       help='每次MCTS迭代数')
    parser.add_argument('--val-interval', type=int, default=5000,
                       help='验证间隔')
    parser.add_argument('--max-iter', type=int, default=50000,
                       help='最大迭代数')
    parser.add_argument('--no-resume', action='store_true',
                       help='禁用自动恢复（从头开始训练）')
    parser.add_argument('--reset', action='store_true',
                       help='重置所有检查点和恢复状态（重新开始）')
    # 多轮闭环优化参数（默认从config读取）
    parser.add_argument('--n-rounds', type=int, 
                       default=3,
                       help="MCTS-验证-重训的循环轮数（默认3轮）")
    parser.add_argument('--top-n-final', type=int, 
                       default=20,
                       help="每轮最终输出并验证的候选数量（默认20个）")
    parser.add_argument('--skip-vina', action='store_true',
                       help='跳过Vina验证（仅用于测试）')
    parser.add_argument('--convergence-metric', type=str, 
                       default='best',
                       choices=['best', 'mean', 'top3_mean', 'median'],
                       help="收敛判定指标（默认best）")
    parser.add_argument('--min-improvement', type=float, 
                       default=0.5,
                       help="最小改善阈值（kcal/mol，默认0.5）")
    parser.add_argument('--patience', type=int, 
                       default=2,
                       help="收敛容忍轮数（默认2）")
    
    # 【新增】迭代闭环优化模式参数
    parser.add_argument('--iterative-loop', action='store_true',
                       help='使用迭代闭环优化模式（新流程：随机补全+批量Vina+EGNN微调）')
    
    args = parser.parse_args()
    
    # 处理重置请求
    if args.reset:
        checkpoint_dir = config.BASE_DIR / "checkpoints" / args.target
        resume_file = checkpoint_dir / "resume_state.json"
        
        print("="*60)
        print("重置训练状态")
        print("="*60)
        
        if resume_file.exists():
            resume_file.unlink()
            print(f"  删除: {resume_file}")
        
        # 删除检查点文件
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.json"):
                f.unlink()
                print(f"  删除: {f}")
        
        print("  训练状态已重置")
        print("="*60)
    
    # 创建引擎（自动恢复）
    engine = Phase2Engine(
        target_name=args.target,
        resume=not args.no_resume
    )
    
    if args.cold_start:
        engine.cold_start(n_sequences=args.n_sequences)
    
    try:
        # 根据n_rounds选择运行模式
        if args.n_rounds > 1:
            # 多轮MCTS-验证-重训闭环（支持自动收敛）
            print(f"\n启动多轮闭环优化: {args.n_rounds} 轮")
            
            # 构建收敛配置
            convergence_config = {
                'patience': args.patience,
                'min_improvement': args.min_improvement,
                'window_size': 3,
                'metric': args.convergence_metric,
                'max_rounds': args.n_rounds
            }
            
            engine.run_multi_round(
                n_rounds=args.n_rounds,
                n_mcts_iterations=args.mcts_iter,
                validation_interval=args.val_interval,
                max_iterations=args.max_iter,
                cold_start_n=args.n_sequences,
                top_n_final=args.top_n_final,
                skip_vina=args.skip_vina,
                convergence_config=convergence_config
            )
        elif args.iterative_loop:
            # 【新增】迭代闭环优化模式（新流程）
            print(f"\n启动迭代闭环优化模式（新流程）")
            engine.run_iterative_loop(max_iterations=config.MAX_EGNN_ITERATIONS)
        else:
            # 单轮MCTS搜索（兼容旧模式）
            engine.run(
                n_mcts_iterations=args.mcts_iter,
                validation_interval=args.val_interval,
                max_iterations=args.max_iter,
                cold_start_n=args.n_sequences
            )
    except KeyboardInterrupt:
        print("\n训练已中断，可以使用相同命令恢复")
        # 关闭日志
        if close_logger:
            close_logger()
        sys.exit(0)
    finally:
        # 确保日志被关闭
        if close_logger:
            close_logger()


if __name__ == "__main__":
    main()
