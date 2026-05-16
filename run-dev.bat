@echo off
setlocal
set "ROOT=%~dp0"

echo ========================================
echo SLI Dev Launcher (backend + frontend)
echo ========================================
echo Note: activate your venv first so Python matches your backend deps.
echo.

call :stop_existing

if not exist "%ROOT%backend\model_v2.onnx" (
    echo [WARNING] backend\model_v2.onnx not found.
    echo Run run.bat option 4 to train or copy a model.
    echo.
)

if not exist "%ROOT%frontend\node_modules" (
    echo [WARNING] frontend\node_modules not found.
    echo Run run.bat option 2 first.
    echo.
)

echo This will open two windows:
echo   - SLI Backend (FastAPI)
echo   - SLI Frontend (Vite)
echo.
echo To stop: press Ctrl+C in each window or close the windows.
echo Re-run will close any previous SLI dev windows first.
echo.

start "SLI Backend" cmd /k "cd /d ""%ROOT%"" ^& python backend\main.py"
start "SLI Frontend" cmd /k "cd /d ""%ROOT%frontend"" ^& npm run dev"

endlocal
exit /b 0

:stop_existing
taskkill /F /T /FI "WINDOWTITLE eq SLI Backend" >nul 2>nul
taskkill /F /T /FI "WINDOWTITLE eq SLI Frontend" >nul 2>nul
exit /b 0
