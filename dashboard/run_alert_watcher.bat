@echo off
REM ==========================================================================
REM  FS50 headless alert watcher.
REM  Runs the Teams alert dispatch 24/7 WITHOUT needing a browser tab open.
REM  Point Windows Task Scheduler at this file: trigger "At log on",
REM  and on the Settings tab tick "If the task fails, restart every 1 minute".
REM
REM  It loops internally (every 5 min, 30-min window) so it only needs to be
REM  STARTED once and kept alive.
REM ==========================================================================
setlocal
set CONDA_ACT=%USERPROFILE%\AppData\Local\miniconda3\Scripts\activate.bat

call "%CONDA_ACT%" fs50defect
cd /d "%~dp0"

python watch_alerts.py --interval 300 --window 30

endlocal
