@echo off
chcp 65001 >nul
cd /d "%~dp0"
python change_history_viewer.py
pause

