@echo off
chcp 65001 >nul
echo ========================================
echo 快速测试模式（用于验证代码修改）
echo ========================================
echo.

REM 设置靶点名称
set TARGET=1LYZ

echo 靶点: %TARGET%
echo.
echo [警告] 此模式使用最小参数快速运行，仅用于测试！
echo.

REM 检查阶段一是否完成
echo [检查] 检查阶段一输出...
if not exist "D:\code\AA\results\%TARGET%\vina\vina-receptor.pdbqt" (
    echo [错误] 阶段一未完成！请先运行阶段一。
    pause
    exit /b 1
)
echo      阶段一已完成
echo.

echo ========================================
echo 快速测试参数:
echo ========================================
echo   - 冷启动序列: 10个（正常: 50-100）
echo   - MCTS迭代: 100次（正常: 1000+）
echo   - 最大迭代: 500次（正常: 10000+）
echo   - 验证间隔: 250次（正常: 5000）
echo   - 最终候选: 5个（正常: 20）
echo   - 跳过Vina验证（仅测试EGNN流程）
echo ========================================
echo.

python run_phase2.py -t %TARGET% ^
    --n-rounds 1 ^
    --n-sequences 10 ^
    --mcts-iter 100 ^
    --max-iter 500 ^
    --top-n-final 5 ^
    --val-interval 250 ^
    --skip-vina

if errorlevel 1 (
    echo [错误] 测试执行失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 快速测试完成！
echo ========================================
echo.
echo 注意: 此模式跳过了Vina验证，仅测试EGNN流程
echo       正式运行请使用 run_phase2_only.bat
echo.
pause
