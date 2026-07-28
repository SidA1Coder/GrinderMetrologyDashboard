@echo off
REM ==========================================================================
REM  FS50 dashboard launcher (Streamlit UI).
REM  Double-click to run, or point Windows Task Scheduler at this file to start
REM  the dashboard automatically at logon.
REM
REM  Assumes Miniconda is installed per-user with a "fs50defect" env.
REM  If the env lives elsewhere, edit CONDA_ACT below.
REM ==========================================================================
setlocal
set CONDA_ACT=%USERPROFILE%\AppData\Local\miniconda3\Scripts\activate.bat

call "%CONDA_ACT%" fs50defect
cd /d "%~dp0"

REM --server.address 0.0.0.0 lets teammates open it from their own PCs.
python -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501

endlocal
