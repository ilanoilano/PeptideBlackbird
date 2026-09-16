@echo off
chcp 65001 >nul
echo ========================================
echo 清理 MCTS 残留文件
echo ========================================
echo.

REM 清理日志文件
echo [1/5] 清理日志文件...
if exist "D:\code\AA\logs\*.log" (
    del /Q "D:\code\AA\logs\*.log"
    echo      已删除日志文件
) else (
    echo      无日志文件需要清理
)

REM 清理检查点
echo [2/5] 清理检查点文件...
if exist "D:\code\AA\checkpoints" (
    rmdir /S /Q "D:\code\AA\checkpoints"
    echo      已删除检查点目录
    mkdir "D:\code\AA\checkpoints"
) else (
    echo      无检查点需要清理
)

REM 清理临时文件
echo [3/5] 清理临时文件...
if exist "D:\code\AA\temp" (
    rmdir /S /Q "D:\code\AA\temp"
    echo      已删除临时目录
    mkdir "D:\code\AA\temp"
) else (
    echo      无临时文件需要清理
)

REM 清理结果目录（可选，保留PDB文件）
echo [4/5] 清理结果目录中的CSV文件...
if exist "D:\code\AA\results\1LYZ\*.csv" (
    del /Q "D:\code\AA\results\1LYZ\*.csv"
    echo      已删除CSV结果文件
)

REM 清理恢复状态
echo [5/5] 清理恢复状态文件...
if exist "D:\code\AA\checkpoints\1LYZ\resume_state.json" (
    del /Q "D:\code\AA\checkpoints\1LYZ\resume_state.json"
    echo      已删除恢复状态
)

echo.
echo ========================================
echo 清理完成！
echo ========================================
pause
