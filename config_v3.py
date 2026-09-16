# -*- coding: utf-8 -*-
"""
MCTS-EGNN闭环优化配置 V3（精简版）

只保留V3版本实际使用的配置
"""

from pathlib import Path

# =============================================================================
# 项目路径配置
# =============================================================================
BASE_DIR = Path("/mnt/d/code/AA")
PDB_DIR = BASE_DIR / "PDB"
RESULTS_DIR = BASE_DIR / "results"
FINAL_DIR = BASE_DIR / "final"
MODELS_DIR = BASE_DIR / "models"

# 子目录路径（动态生成，基于靶点名称）
def get_target_dirs(target_name: str) -> dict:
    """获取指定靶点的所有工作目录"""
    target_base = RESULTS_DIR / target_name
    return {
        "cleaned": target_base / "cleaned",
        "pocket": target_base / "pocket",
        "vina": target_base / "vina",
        "af3_receptor": target_base / "AF3_receptor",
    }


# 靶点特定的 pocket 文件名配置
TARGET_POCKET_FILES = {
    "1LYZ": "pocket-for-esmif.pdb",
}


def get_pocket_pdb_path(target_name: str) -> Path:
    """获取指定靶点的 pocket PDB 文件路径"""
    pocket_dir = get_target_dirs(target_name)["pocket"]
    pocket_filename = TARGET_POCKET_FILES.get(target_name, "pocket1.pdb")
    return pocket_dir / pocket_filename


# =============================================================================
# 肽模板配置
# =============================================================================

# 肽序列模板
PEPTIDE_TEMPLATE = "ACxxxxxxCxxxxxxCG"

# 固定位置映射：{位置索引: 氨基酸}
FIXED_POSITIONS = {
    0: "A",   # N端
    1: "C",   # 第一个半胱氨酸
    8: "C",   # 第二个半胱氨酸
    15: "C",   # 第三个半胱氨酸
    16: "G"   # C端
}

# 可变位置（模板中 'x' 的位置）
def get_variable_positions(template: str = None) -> list:
    """从模板中提取可变位置索引"""
    if template is None:
        template = PEPTIDE_TEMPLATE
    return [i for i, aa in enumerate(template) if aa == 'x']

VARIABLE_POSITIONS = get_variable_positions()

# 标准20种氨基酸（排除Cys用于可变位置）
ALLOWED_AMINO_ACIDS = [
    "A", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]

# 每个可变位置的可选氨基酸
VARIABLE_AMINO_ACIDS = {
    pos: ALLOWED_AMINO_ACIDS.copy() for pos in VARIABLE_POSITIONS
}


# =============================================================================
# 环化配置
# =============================================================================

# 化学交联剂配置
CROSSLINKER = "TBMB"

# 交联剂连接位置（TBMB需要3个Cys）
CROSSLINKER_POSITIONS = [1,8,15]


# =============================================================================
# 自适应MCTS-EGNN闭环优化配置（V3版本）
# =============================================================================

# EGNN迭代轮次相关的动态函数
# N: 当前EGNN迭代轮次（从1开始）

def f_n(n: int) -> int:
    """
    选取母节点数量（随EGNN轮次递增）
    第1轮: 19个，之后每轮+2，上限100
    """
    return min(19 + (n - 1) ** 2, 300)

def g_n(n: int) -> int:
    """
    每个母节点的随机填充数（随EGNN轮次递减）
    第1轮: 50个，之后每轮-2，下限10
    """
    return max(26 - (n - 1) * 2, 14)

def h_n(n: int) -> int:
    """
    Vina验证数量（随EGNN轮次递增）
    第1轮: 40个，之后每轮+3，上限200
    """
    return min(40 + (n - 1) * 12, 300)

# Softmax分配配置
SOFTMAX_TEMPERATURE = 1.0       # Softmax温度参数
MAX_EXPANSIONS_PER_NODE = 19    # 每个母节点最大扩展数（19种氨基酸，排除Cys）

# 最大EGNN迭代轮数
MAX_EGNN_ITERATIONS = 100
CONVERGENCE_PATIENCE = 3     # 收敛判定耐心值（连续N轮无改善）


# =============================================================================
# MCTS配置
# =============================================================================
MCTS_CONFIG = {
    "c_puct": 0.814,            # PUCT 探索常数
    "max_expansions": 19,       # 每次扩展的最大子节点数
}


# =============================================================================
# EGNN训练配置
# =============================================================================
EGNN_CONFIG = {
    "hidden_dim": 128,          # EGNN 隐藏层维度
    "num_layers": 4,            # EGNN 层数
    "learning_rate": 5e-4,      # 学习率
    "batch_size": 4,           # 批次大小
    "num_epochs": 100,          # 训练轮数
    "patience": 5,              # 早停耐心值
}


# =============================================================================
# Vina 对接配置
# =============================================================================
VINA_CONFIG = {
    "exhaustiveness": 6,        # 搜索详尽度
    "num_modes": 9,             # 输出构象数量
    "energy_range": 4,          # 能量范围（kcal/mol）
    "cpu": 8,                   # CPU核心数
}
# =============================================================================
# Vina 对接验证配置
# =============================================================================
VINA_VALIDATION = {
    "enable": True,              # 是否启用验证
    "max_distance": 7,         # 质心到口袋最大距离 (Å)
    "min_atoms": 30,             # 最小原子数
    "max_atoms": 2000,            # 最大原子数
    "min_energy": -15.0,         # 最小结合能
    "max_energy": -3.0,          # 最大结合能
}

# 对接盒子默认尺寸（Å）
VINA_BOX_SIZE = (20, 20,20)


# =============================================================================
# 外部工具路径配置
# =============================================================================
TOOLS = {
    "fpocket": "fpocket",
    "obabel": "/usr/bin/obabel",
    "vina": "/home/ilano/miniconda3/envs/AA/bin/vina",
    "babel_libdir": "/home/ilano/miniconda3/envs/AA/lib/openbabel/2.4.1",
}


# =============================================================================
# 氨基酸物理化学性质（用于启发式评分）
# =============================================================================

# Kyte-Doolittle 疏水性标度
HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5,
    "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# 氨基酸分子量（Da）
AA_MOLECULAR_WEIGHT = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13,
    "F": 165.19, "G": 75.07, "H": 155.16, "I": 131.17,
    "K": 146.19, "L": 131.17, "M": 149.21, "N": 132.12,
    "P": 115.13, "Q": 146.15, "R": 174.20, "S": 105.09,
    "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}
