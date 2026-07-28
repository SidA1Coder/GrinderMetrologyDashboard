@echo off
REM ==========================================================================
REM  FS50 dashboard launcher (Streamlit UI).
REM  Double-click to run, or point Windows Task Scheduler at this file to start
REM  the dashboard automatically at logon.
REM
REM  Assumes Miniconda is installed per-user with a "fs50defect" env.
REM  If the env lives elsewhere, edit ENV_PY below.
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

REM Change PORT here to run the dashboard on a different port (e.g. 80, 8080).
set PORT=8502

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

REM Stop any dashboard already listening on this port so this launch always uses
REM the latest code and .env (otherwise the old process keeps the port and you
REM see stale data).
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
    echo Stopping old dashboard (PID %%P)...
    taskkill /PID %%P /F >nul 2>&1
)

echo Starting FS50 dashboard on port %PORT% ...
echo Local:   http://localhost:%PORT%
echo Network: http://%COMPUTERNAME%:%PORT%
echo.

REM --server.address 0.0.0.0 lets teammates open it from their own PCs.
"%ENV_PY%" -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port %PORT%

echo.
echo Dashboard stopped (exit code %ERRORLEVEL%).
pause

endlocal
