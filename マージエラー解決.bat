@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   マージエラー解決スクリプト
echo ========================================
echo.
echo 問題のあるファイルの変更を一時的に保存します...
echo.

REM 変更を一時的に保存
git stash push -m "マージ前の一時保存" old/inspection_focus.py old/inspection_perfect.py old/inspection_ultra_fixed.py old/trainer_final.py scripts/train_4class_sparse_ensemble.py

if %ERRORLEVEL% NEQ 0 (
    echo 変更を保存できませんでした。
    echo 別の方法を試します...
    echo.
    
    REM 変更を破棄する場合（注意：変更が失われます）
    echo これらのファイルの変更を破棄しますか？
    echo 変更を保持したい場合は、このスクリプトを中断してください。
    pause
    
    git checkout -- old/inspection_focus.py
    git checkout -- old/inspection_perfect.py
    git checkout -- old/inspection_ultra_fixed.py
    git checkout -- old/trainer_final.py
    git checkout -- scripts/train_4class_sparse_ensemble.py
)

echo.
echo リモートの変更をマージします...
git pull origin master

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   マージ成功！
    echo ========================================
    echo.
    echo 保存した変更を復元します...
    git stash pop
    echo.
    echo プッシュを実行します...
    git push origin master
) else (
    echo.
    echo ========================================
    echo   マージに失敗しました。
    echo ========================================
    echo.
    echo 保存した変更を復元します...
    git stash pop
)

echo.
pause


