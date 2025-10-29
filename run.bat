@echo off
REM Sign Language Recognition - Complete Project Launcher
REM Quick start script for Windows

echo ========================================
echo Sign Language Recognition - SLI
echo ========================================
echo.

:menu
echo What would you like to do?
echo.
echo === SETUP ===
echo 1. Install dependencies (backend)
echo 2. Install frontend dependencies
echo 4. Train model
echo 5. Test model (single image)
echo 6. Test model (random samples)
echo 7. Start FastAPI server
echo 8. Test API endpoints
echo 9. Start React frontend (dev)
echo 10. Build frontend (production)
echo 11. View model info
echo 12. Open API documentation
echo 13. Preprocess dataset (MediaPipe Hands)
echo 14. Evaluate model (H5/ONNX)
echo 15. Run end-to-end pipeline test
echo 0. Exit
echo.

set /p choice="Enter your choice (0-15): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto install_frontend
if "%choice%"=="4" goto train
if "%choice%"=="5" goto test_single
if "%choice%"=="6" goto test_samples
if "%choice%"=="7" goto start_server
if "%choice%"=="8" goto test_api
if "%choice%"=="9" goto start_frontend
if "%choice%"=="10" goto build_frontend
if "%choice%"=="11" goto model_info
if "%choice%"=="12" goto api_docs
if "%choice%"=="13" goto preprocess
if "%choice%"=="14" goto evaluate
if "%choice%"=="15" goto pipeline_test
if "%choice%"=="0" goto exit

echo Invalid choice. Please try again.
echo.
goto menu

:install
echo.
echo Installing backend dependencies...
echo ========================================
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo ✓ Backend dependencies installed!
echo.
pause
goto menu

:install_frontend
echo.
echo Installing frontend dependencies...
echo ========================================
cd frontend
call npm install
cd ..
echo.
echo ✓ Frontend dependencies installed!
echo.
pause
goto menu


:train
echo.
echo Starting model training...
echo ========================================
echo This may take 30-60 minutes with GPU
echo or 2-4 hours with CPU.
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto menu
echo.
python ML/train_optimized.py
echo.
echo ✓ Training completed!
echo.
pause
goto menu

:test_single
echo.
echo Test on specific image
echo ========================================
set /p image_path="Enter image path (e.g., ./data/stop/1.png): "
echo.
python ML/inference.py "%image_path%"
echo.
pause
goto menu

:test_samples
echo.
echo Testing on random samples...
echo ========================================
python ML/inference.py
echo.
pause
goto menu

:model_info
echo.
echo Model Information
echo ========================================
if exist "backend\model_v2.onnx" (
    python backend\onnx_utils.py
) else (
    echo model_v2.onnx not found. Please train the model first.
    echo Run option 4: Train model
)
echo.
pause
goto menu

:start_server
echo.
echo Starting FastAPI Server...
echo ========================================
if not exist "backend\model_v2.onnx" (
    echo [WARNING] model_v2.onnx not found!
    echo Please train the model first (option 3^)
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" goto menu
)
echo.
echo Server will be available at:
echo   - API:   http://localhost:8000
echo   - Docs:  http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
python backend\main.py
pause
goto menu

:test_api
echo.
echo Testing API Endpoints
echo ========================================
echo Make sure the server is running first!
echo (Run option 7: Start FastAPI server in another window)
echo.
set /p continue="Continue with API tests? (y/n): "
if /i not "%continue%"=="y" goto menu
echo.
python backend\test_api.py
echo.
pause
goto menu

:start_frontend
echo.
echo Starting React Frontend (Development Mode)
echo ========================================
echo.
echo Frontend will be available at:
echo   http://localhost:3000
echo.
echo NOTE: Make sure the backend is running on port 8000!
echo (Run option 7 in another window)
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
cd frontend
call npm run dev
cd ..
pause
goto menu

:build_frontend
echo.
echo Building Frontend for Production
echo ========================================
echo.
cd frontend
call npm run build
echo.
echo ✓ Build complete! Output in frontend/dist/
echo.
echo To preview production build:
call npm run preview
cd ..
echo.
pause
goto menu

:preprocess
echo.
echo Preprocess dataset using MediaPipe Hands
echo ========================================
echo.
set /p src="Source directory (default: data): "
if "%src%"=="" set src=data
set /p dst="Destination directory (default: data_preprocessed): "
if "%dst%"=="" set dst=data_preprocessed
echo.
python ML\preprocess_hands.py --src "%src%" --dst "%dst%"
echo.
pause
goto menu

:evaluate
echo.
echo Evaluate trained model (Keras H5 or ONNX)
echo ========================================
echo.
set /p model="Model path (default: backend\model_v2.onnx): "
if "%model%"=="" set model=backend\model_v2.onnx
set /p data="Dataset directory (default: data): "
if "%data%"=="" set data=data
set /p outdir="Output directory (default: evaluation): "
if "%outdir%"=="" set outdir=evaluation
echo.
python ML\evaluate_model.py --model "%model%" --data "%data%" --out "%outdir%"
echo.
pause
goto menu

:pipeline_test
echo.
echo End-to-End Pipeline Test
echo ========================================
echo This will:
echo   1. Start FastAPI backend
echo   2. Build and start React frontend
echo   3. Test all API endpoints
echo   4. Run model evaluation
echo   5. Generate test report
echo.
echo NOTE: This will take 2-5 minutes to complete.
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto menu
echo.
python test_pipeline.py
echo.
pause
goto menu

REM (duplicate model_info removed)

:api_docs
echo.
echo Opening API Documentation...
echo ========================================
echo.
echo Make sure the server is running (option 7)
echo.
start http://localhost:8000/docs
echo ✓ Opening browser with API documentation
echo.
pause
goto menu

:exit
echo.
echo Thank you for using Sign Language Recognition!
echo.
exit
