@echo off
REM Sign Language Recognition - Complete Project Launcher
REM Quick start script for Windows
REM Full setup + testing: see docs\ONBOARDING_AND_IMPLEMENTATION_GUIDE.md

echo ========================================
echo Sign Language Recognition - SLI
echo ========================================
echo Docs: docs\ONBOARDING_AND_IMPLEMENTATION_GUIDE.md
echo.

:menu
echo What would you like to do?
echo.
echo === SETUP ===
echo 1. Install dependencies (backend)
echo 2. Install frontend dependencies
echo 3. Sync training data from Hugging Face into data/
echo === TRAINING ===
echo 4. Train model (EfficientNetV2-S - recommended)
echo 17. Train model (MobileNetV3-Large - legacy)
echo 18. Download ISL phrase datasets (Mendeley)
echo 19. Ingest external images into data/ (merge new dataset)
echo === INFERENCE / UI ===
echo 5. Test model (single image)
echo 6. Test model (random samples)
echo 7. Start FastAPI server (REST + optional WebSocket /ws for WebRTC)
echo 8. Test API endpoints
echo 9. Start React frontend (dev)
echo 10. Build frontend (production)
echo 11. View model info
echo 12. Open API documentation
echo === OPTIONAL WEBRTC ===
echo 16. Install WebRTC deps (requirements-webrtc.txt, needs same Python as option 1)
echo === OTHER ===
echo 13. Preprocess dataset (MediaPipe Hands)
echo 14. Evaluate model (H5/ONNX)
echo 15. Run end-to-end pipeline test
echo 0. Exit
echo.

set /p choice="Enter your choice (0-19): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto install_frontend
if "%choice%"=="3" goto pull_hf_data
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
if "%choice%"=="16" goto install_webrtc
if "%choice%"=="17" goto train_mobilenet
if "%choice%"=="18" goto download_isl
if "%choice%"=="19" goto ingest_external
if "%choice%"=="0" goto exit

echo Invalid choice. Please try again.
echo.
goto menu

:pull_hf_data
echo.
echo Syncing training images from Hugging Face (dataset repo into data/)
echo ========================================
echo Uses huggingface_hub; run option 1 first if dependencies are missing.
echo Override repo with: set SLI_HF_DATASET_REPO=namespace/name
echo.
python ML\pull_data_from_hf.py
echo.
pause
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

:install_webrtc
echo.
echo Installing optional WebRTC dependencies (aiortc^)
echo ========================================
echo Use the SAME Python environment as backend option 1 (venv recommended^).
echo.
pip install -r requirements-webrtc.txt
echo.
echo ✓ WebRTC extras installed. Restart the API (option 7^) for /ws/webrtc.
echo.
pause
goto menu


:train
echo.
echo Training EfficientNetV2-S (ImageNet weights, mixed precision on GPU)
echo ========================================
echo This uses data/ (sync from Hugging Face with option 3, or local / ingested images).
echo Estimated time:
echo   - CUDA GPU (e.g. RTX 3050+): ~30-90 minutes depending on dataset size
echo   - CPU only: several hours
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto menu
echo.
python ML/train_efficientnetv2.py
echo.
echo Training completed. Outputs in backend/: model_v2.onnx, best_model.h5, class_labels.txt
echo.
pause
goto menu

:train_mobilenet
echo.
echo Training MobileNetV3-Large (legacy, smaller/faster but less accurate)
echo ========================================
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto menu
echo.
python ML/train_mobilenet.py
echo.
pause
goto menu

:download_isl
echo.
echo Downloading Indian Sign Language phrase datasets (CC BY 4.0)
echo ========================================
echo   phrases_v2     : Mendeley w7fgy7jvs8 v2  (44 classes x 40 images)
echo   common_phrases : Mendeley y8vg69brn2     (40 classes x 30 images)
echo   all            : both of the above
echo.
set /p which="Which dataset? (phrases_v2 / common_phrases / all): "
if "%which%"=="" set which=all
echo.
python ML/download_isl_phrases.py --dataset "%which%"
echo.
echo Next: inspect datasets_raw/, then use option 19 to merge images into data/.
pause
goto menu

:ingest_external
echo.
echo Ingest external images into data/ (merge new dataset with existing classes)
echo ========================================
set /p src="Source directory (e.g. datasets_raw\phrases_v2): "
set /p mapping="Mapping YAML (default: ML\external_gloss_mapping.example.yaml): "
if "%mapping%"=="" set mapping=ML\external_gloss_mapping.example.yaml
set /p cap="Max images per class (default: 200): "
if "%cap%"=="" set cap=200
echo.
python ML/ingest_external.py --mode local_images --src "%src%" --mapping "%mapping%" --max-per-class %cap%
echo.
echo When satisfied, retrain with option 4 (EfficientNetV2-S).
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
    echo Please train the model first (option 4^)
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
