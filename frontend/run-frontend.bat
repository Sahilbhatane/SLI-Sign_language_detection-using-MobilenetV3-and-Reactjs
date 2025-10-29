@echo off
echo ========================================
echo   Sign Language Interpreter - Frontend
echo ========================================
echo.

cd frontend

:menu
echo.
echo Select an option:
echo.
echo 1. Install dependencies (npm install)
echo 2. Start development server (npm run dev)
echo 3. Build for production (npm run build)
echo 4. Preview production build (npm run preview)
echo 5. Open in browser
echo 0. Exit
echo.

set /p choice="Enter your choice (0-5): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto dev
if "%choice%"=="3" goto build
if "%choice%"=="4" goto preview
if "%choice%"=="5" goto browser
if "%choice%"=="0" goto end

echo Invalid choice. Please try again.
goto menu

:install
echo.
echo Installing dependencies...
call npm install
echo.
echo Dependencies installed successfully!
pause
goto menu

:dev
echo.
echo Starting development server...
echo Frontend will be available at http://localhost:3000
echo Press Ctrl+C to stop the server
echo.
call npm run dev
pause
goto menu

:build
echo.
echo Building for production...
call npm run build
echo.
echo Build complete! Output in dist/ folder
pause
goto menu

:preview
echo.
echo Starting production preview...
echo Preview will be available at http://localhost:4173
echo Press Ctrl+C to stop the server
echo.
call npm run preview
pause
goto menu

:browser
echo.
echo Opening http://localhost:3000 in browser...
start http://localhost:3000
goto menu

:end
echo.
echo Goodbye!
echo.
cd ..
