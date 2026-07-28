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

REM Full path to the Python interpreter inside the fs50defect conda env.
REM Using it directly avoids conda's activate.bat, which can close this window.
REM The env location differs per machine, so we try several common spots.
set ENV_PY=
for %%D in (
    "%USERPROFILE%\AppData\Local\miniconda3\envs\fs50defect\python.exe"
    "%USERPROFILE%\miniconda3\envs\fs50defect\python.exe"
    "%USERPROFILE%\AppData\Local\Anaconda3\envs\fs50defect\python.exe"
    "%USERPROFILE%\Anaconda3\envs\fs50defect\python.exe"
    "C:\New folder\envs\fs50defect\python.exe"
    "C:\ProgramData\miniconda3\envs\fs50defect\python.exe"
    "C:\ProgramData\Anaconda3\envs\fs50defect\python.exe"
) do (
    if exist %%D set "ENV_PY=%%~D"
)

cd /d "%~dp0"

if not defined ENV_PY (
    echo.
    echo [ERROR] Could not find the fs50defect Python in any known location.
    echo Find it by running in an Anaconda Prompt:
    echo     conda activate fs50defect ^&^& where python
    echo Then add that full path to the list at the top of this file.
    echo.
    pause
    exit /b 1
)

echo Using Python: "%ENV_PY%"
echo Starting FS50 alert watcher (5-min loop, 30-min window)...
echo.

"%ENV_PY%" watch_alerts.py --interval 300 --window 30

echo.
echo Alert watcher stopped (exit code %ERRORLEVEL%).
pause

endlocal
