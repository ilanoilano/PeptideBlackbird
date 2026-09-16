@echo off
chcp 65001 >nul
echo ========================================
echo 仅运行阶段二: MCTS搜索
echo ========================================
echo.

REM 设置靶点名称（可修改）
set TARGET=1LYZ

echo 靶点: %TARGET%
echo.

REM 检查阶段一是否完成
echo [检查] 检查阶段一输出...
if not exist "D:\code\AA\results\%TARGET%\vina\vina-receptor.pdbqt" (
    echo [错误] 阶段一未完成！请先运行阶段一。
    echo        缺失文件: results\%TARGET%\vina\vina-receptor.pdbqt
    pause
    exit /b 1
)
echo      阶段一已完成，可以继续
echo.

REM 检查是否已存在EGNN模型
echo [检查] 检查EGNN模型状态...
if exist "D:\code\AA\egnn\models\best_model.pt" (
    echo      发现已有EGNN模型
) else (
    echo      未找到EGNN模型，阶段二将执行冷启动
)
echo.

echo ========================================
echo 选择运行模式:
echo ========================================
echo.
echo [1] 多轮MCTS闭环优化（推荐，3轮）
echo [2] 单轮MCTS搜索
echo [3] 迭代闭环优化模式（新流程）
echo.
set /p MODE="请选择模式 (1/2/3): "

echo.
echo ========================================

if "%MODE%"=="1" (
    echo 运行模式: 多轮MCTS闭环优化
echo.
    python run_phase2.py -t %TARGET% ^
        --n-rounds 3 ^
        --n-sequences 50 ^
        --mcts-iter 1000 ^
        --max-iter 10000 ^
        --top-n-final 20 ^
        --val-interval 5000
) else if "%MODE%"=="2" (
    echo 运行模式: 单轮MCTS搜索
echo.
    python run_phase2.py -t %TARGET% ^
        --n-sequences 50 ^
        --mcts-iter 1000 ^
        --max-iter 10000 ^
        --val-interval 5000 ^
        --no-resume
) else if "%MODE%"=="3" (
    echo 运行模式: 迭代闭环优化（新流程）
echo.
    python run_phase2.py -t %TARGET% ^
        --iterative-loop ^
        --n-sequences 50
) else (
    echo [错误] 无效的选择！
    pause
    exit /b 1
)

if errorlevel 1 (
    echo [错误] 阶段二执行失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 阶段二执行完成！
echo ========================================
echo.
echo 结果位置:
echo   - 候选结果: D:\code\AA\results\%TARGET%\candidates.csv
echo   - 数据集: D:\code\AA\results\%TARGET%\dataset.csv
echo   - 日志: D:\code\AA\logs\
echo   - 检查点: D:\code\AA\checkpoints\%TARGET%\
echo.
pause
