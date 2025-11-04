@echo off
echo Running one-time GitHub sync...
echo This will check for changes and commit if threshold is reached
echo.
python integrated_github_system.py once
pause
