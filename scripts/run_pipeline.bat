@echo off
chcp 65001 >nul
echo ========================================
echo MCTS 环肽设计流水线
echo ========================================
echo.

REM 设置靶点名称（可修改）
set TARGET=1LYZ

echo 靶点: %TARGET%
echo.

REM 检查是否已存在EGNN模型
echo [检查] 检查EGNN模型状态...
if exist "D:\code\AA\egnn\models\best_model.pt" (
    echo      发现已有EGNN模型，将跳过冷启动
    set SKIP_COLD_START=--no-resume
) else (
    echo      未找到EGNN模型，将执行冷启动
    set SKIP_COLD_START=
)

echo.
echo ========================================
echo 阶段一: PDB预处理
echo ========================================
echo.

python run_phase1.py -t %TARGET% --pdb PDB\%TARGET%.pdb\pdb%TARGET%.ent

if errorlevel 1 (
    echo [错误] 阶段一执行失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 阶段二: MCTS搜索
echo ========================================
echo.

REM 多轮MCTS闭环优化模式（推荐）
echo 运行模式: 多轮MCTS闭环优化
echo.

python run_phase2.py -t %TARGET% ^
    --n-rounds 3 ^
    --n-sequences 50 ^
    --mcts-iter 1000 ^
    --max-iter 10000 ^
    --top-n-final 20 ^
    --val-interval 5000

if errorlevel 1 (
    echo [错误] 阶段二执行失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 流水线执行完成！
echo ========================================
echo.
echo 结果位置:
echo   - 候选结果: D:\code\AA\results\%TARGET%\candidates.csv
echo   - 数据集: D:\code\AA\results\%TARGET%\dataset.csv
echo   - 日志: D:\code\AA\logs\
echo   - 检查点: D:\code\AA\checkpoints\%TARGET%\
echo.
pause
