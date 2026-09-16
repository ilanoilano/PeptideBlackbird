@echo off
chcp 65001 >nul
echo ========================================
echo   安装 OpenBabel 和 PDBFixer
echo ========================================
echo.

REM 检查 WSL
wsl --version >nul 2>&1
if errorlevel 1 (
    echo [错误] WSL 未安装或未启用
    echo 请参考: https://docs.microsoft.com/zh-cn/windows/wsl/install
    pause
    exit /b 1
)

echo [OK] WSL 已安装
echo.

REM 复制脚本到 WSL
echo 复制安装脚本到 WSL...
copy /Y "%~dp0install_deps.sh" "\\wsl$\Ubuntu\tmp\install_deps.sh" >nul 2>&1
if errorlevel 1 (
    echo 尝试使用 wsl 命令复制...
    wsl -e cp "/mnt/d/code/AA/install_deps.sh" "/tmp/install_deps.sh"
)

REM 在 WSL 中执行安装
echo.
echo 在 WSL 中执行安装...
echo 可能需要输入 sudo 密码
echo.
wsl -e bash /tmp/install_deps.sh

if errorlevel 1 (
    echo.
    echo [错误] 安装失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
pause
