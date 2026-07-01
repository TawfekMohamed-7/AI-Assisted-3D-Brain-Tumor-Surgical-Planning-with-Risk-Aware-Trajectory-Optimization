@echo off
setlocal

:: Resolve project root from this file's location
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

echo.
echo  ============================================
echo   Brain Tumor Pre-Surgery Planning System
echo  ============================================
echo   Project: %ROOT%
echo.

:: [1] Install backend Python dependencies (quiet)
echo  [1/3] Checking backend dependencies...
python -m pip install -r requirements.txt -q --no-warn-script-location
echo  Backend dependencies OK.
echo.

:: [2] Install frontend Node dependencies if node_modules is missing
set "FRONTEND=%ROOT%\Smart-Pre-Surgery-Planning-master"
if not exist "%FRONTEND%\package.json" set "FRONTEND=%ROOT%\frontend"

if exist "%FRONTEND%\package.json" (
    if not exist "%FRONTEND%\node_modules" (
        echo  [2/3] Installing frontend dependencies...
        pushd "%FRONTEND%"
        npm install --legacy-peer-deps
        popd
    ) else (
        echo  [2/3] Frontend dependencies already installed.
    )
) else (
    echo  [2/3] No frontend folder found.
    set "FRONTEND="
)
echo.

:: [3] Free ports 8000 and 3000 if already in use
echo  Checking ports...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000') do taskkill /F /PID %%a 2>nul
echo  Ports ready.
echo.
 
:: [4] Start both servers in named windows
echo  [3/3] Starting servers...
echo.
 
:: Backend window
start "Backend | FastAPI :8000" /d "%ROOT%" cmd /k "set PYTHONPATH=%ROOT% && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app"
 
:: Frontend window (only if detected)
if defined FRONTEND (
    if exist "%FRONTEND%\package.json" (
        start "Frontend | React :3000" /d "%FRONTEND%" cmd /k "set BROWSER=none && node node_modules\react-scripts\bin\react-scripts.js start"
    )
)


echo  -------------------------------------------------------------
echo   Backend  ^>  http://localhost:8000
echo   API Docs ^>  http://localhost:8000/docs
if defined FRONTEND echo   Frontend ^>  http://localhost:3000
echo.
echo   Check the two new windows that just opened.
echo   Close them to stop the servers.
echo  -------------------------------------------------------------
echo.
