# 🎯 Sign Language Recognition - Complete Guide

Production-ready sign language phrase recognition system using MobileNetV2 transfer learning and FastAPI backend.

---

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Frontend Application](#-frontend-application)
- [Model Training](#-model-training)
- [FastAPI Backend](#-fastapi-backend)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Technical Specifications](#-technical-specifications)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Dataset (Optional but Recommended)
```bash
python verify_dataset.py
```

### 3. Train the Model
```bash
python train.py
```
**Expected time:** 30-60 minutes (GPU) or 2-4 hours (CPU)

### 4. Test the Model
```bash
# Test on specific image
python inference.py ./data/stop/1.png

# Test on random samples
python inference.py
```

### 5. Start the Backend API
```bash
python backend/main.py
```
**API available at:** http://localhost:8000
**API Docs:** http://localhost:8000/docs

### Alternative: Use the Interactive Launcher
```bash
run.bat
```

---

## 📁 Project Structure

```
SLI/
├── 📄 train.py                    # Main training script
├── 📄 inference.py                # Model testing
├── 📄 verify_dataset.py           # Dataset validation
├── 📄 requirements.txt            # Python dependencies (all)
├── 📄 run.bat                     # Interactive launcher
├── 📄 README.md                   # This file
│
├── 📁 data/                       # Training images (44 classes)
│   ├── 📁 stop/
│   ├── 📁 happy birthday/
│   ├── 📁 wait/
│   └── ... (44 classes total, ~40 images each)
│
├── 📁 backend/
│   ├── 📄 main.py                # FastAPI application
│   ├── 📄 onnx_utils.py          # ONNX wrapper utilities
│   ├── 📄 .env.example           # Environment variables template
│   ├── 🤖 model.onnx             # Trained model (created by training)
│   └── 📄 class_labels.txt       # Class names (created by training)
│
├── 📁 ML/                         # ML experiments
└── 📁 frontend/                   # React frontend application
    ├── src/
    │   ├── components/            # React components
    │   ├── services/              # API & translation services
    │   ├── App.jsx               # Main application
    │   └── index.css             # Tailwind CSS
    ├── package.json
    ├── vite.config.js            # Vite configuration
    └── tailwind.config.js        # Tailwind customization
```

---

## 🎨 Frontend Application

### Overview

Modern React application with:
- ✅ **Real-time Webcam Detection** - Automatic frame capture every 2 seconds
- ✅ **Multi-language Translation** - Hindi, Marathi, Spanish, French, German, Japanese, Chinese, Arabic
- ✅ **Beautiful UI** - TailwindCSS + Framer Motion animations
- ✅ **Reversed L-Shape Layout** - Left vertical navbar + top horizontal header
- ✅ **Detection History Table** - Track all detections with timestamps
- ✅ **Live Backend Monitoring** - Real-time connection status

### Quick Start

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: **http://localhost:3000**

### Features

#### 1. Webcam Capture
- Live webcam feed with frame markers
- Automatic detection every 2 seconds
- Manual capture button
- Recording indicators

#### 2. Detection Display
- Large detected phrase with animations
- Confidence progress bar
- Top 3 alternative predictions
- Detection statistics

#### 3. Language Translation
- Dropdown with 9 languages
- Quick language selection pills
- Automatic translation using LibreTranslate
- Translation loading indicators

#### 4. History Table
- All detections with timestamps
- Confidence scores as progress bars
- Translations for each detection
- Clear history button
- Statistics: total detections, average confidence, unique signs

#### 5. Navigation Tabs
- 🏠 Home - Welcome page with features
- 📹 Detect - Main detection interface
- 📊 History - Detection history table
- 📚 Learn - Learning resources (coming soon)
- ⚙️ Settings - App settings and backend status
- ℹ️ About - Project information

### Technology Stack

- **React** 18.3.1 - UI library
- **Vite** 5.4.11 - Build tool & dev server
- **TailwindCSS** 3.4.0 - Utility-first CSS
- **Framer Motion** 11.0.3 - Smooth animations
- **React-Webcam** 7.2.0 - Webcam access
- **Axios** 1.6.2 - HTTP client for API calls
- **LibreTranslate** - Free translation API

### API Integration

Frontend uses Vite proxy to connect to backend:

```javascript
// Frontend makes requests to /api/*
// Vite proxies to http://localhost:8000/*

axios.post('/api/predict', { image: base64Image })
axios.get('/api/health')
```

### Building for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

Output will be in `frontend/dist/` directory.

---

## 🎓 Model Training

### What You'll Get

After training completes, you'll have:
1. **`backend/model.onnx`** - Production-ready ONNX model
2. **`backend/class_labels.txt`** - List of all 44 sign phrases
3. **`training_history.png`** - Training/validation accuracy & loss plots
4. **`best_model.h5`** - Keras checkpoint (backup)

### Training Process

The script uses a **two-stage training approach**:

#### Stage 1: Initial Training (Frozen Base)
- MobileNetV2 base layers are frozen
- Only custom top layers are trained
- Fast convergence to good baseline

#### Stage 2: Fine-Tuning
- Last 30 layers of MobileNetV2 are unfrozen
- Lower learning rate (10× reduction)
- Refines features for sign language domain

### Configuration

Edit `train.py` to customize:

```python
class Config:
    IMG_HEIGHT = 224          # Image height
    IMG_WIDTH = 224           # Image width
    BATCH_SIZE = 32           # Batch size (reduce if OOM)
    EPOCHS = 50               # Max epochs
    VALIDATION_SPLIT = 0.2    # 20% for validation
    LEARNING_RATE = 0.0001    # Initial learning rate
    PATIENCE = 10             # Early stopping patience
```

### Dataset Verification

Before training, verify your dataset:

```bash
python verify_dataset.py
```

**This checks:**
- ✅ Image count per class
- ✅ Dataset balance
- ✅ Corrupted images
- ✅ Image size consistency
- ✅ Recommendations

### Expected Performance

- **Validation Accuracy**: 85-95%
- **Top-3 Accuracy**: 95-99%
- **Training Time**: 
  - GPU: 30-60 minutes
  - CPU: 2-4 hours

### Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (pre-trained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) + BatchNorm + Dropout(0.5)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(44, softmax)
    ↓
Predictions
```

**Total Parameters:** ~3M
- MobileNetV2: ~2.2M
- Custom layers: ~800K

### Data Augmentation

Training images undergo:
- Rotation: ±20°
- Width/Height shift: ±20%
- Shear: 20%
- Zoom: ±20%
- Horizontal flip
- Rescaling to [0, 1]

---

## 🚀 FastAPI Backend

### Quick Start

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the server
python backend/main.py
```

Server will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check - `GET /health`

Check if the API is ready and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "./backend/model.onnx",
  "num_classes": 44,
  "input_shape": [1, 224, 224, 3]
}
```

#### 2. Predict Sign Language - `POST /predict` ⭐

Predict sign language phrase from base64-encoded image.

**Request:**
```json
{
  "image": "base64_encoded_image_or_data_url",
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "happy birthday",
  "confidence": 94.23,
  "predictions": [
    {
      "rank": 1,
      "class": "happy birthday",
      "confidence": 0.9423,
      "confidence_percent": 94.23
    },
    {
      "rank": 2,
      "class": "congratulations",
      "confidence": 0.0312,
      "confidence_percent": 3.12
    }
  ],
  "processing_time_ms": 45.67
}
```

#### 3. Get All Classes - `GET /classes`

Returns all 44 available sign language classes.

#### 4. Get Model Info - `GET /model-info`

Returns detailed model configuration.

#### 5. Root - `GET /`

Returns API information and available endpoints.

### React Frontend Integration

```javascript
// Convert image to base64 and predict
const handleImageUpload = async (imageFile) => {
  // Convert to base64
  const reader = new FileReader();
  reader.readAsDataURL(imageFile);
  
  reader.onload = async () => {
    const base64Image = reader.result;
    
    // Send to API
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Image,
        top_k: 5
      })
    });
    
    const data = await response.json();
    
    // Display results
    console.log('Prediction:', data.prediction);
    console.log('Confidence:', data.confidence + '%');
    console.log('Processing Time:', data.processing_time_ms + 'ms');
  };
};
```

### Python Client Example

```python
import base64
import requests

# Read and encode image
with open('test_image.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post(
    'http://localhost:8000/predict',
    json={'image': image_data, 'top_k': 5}
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

### CORS Configuration

**Development (Current):**
```python
# Allows all origins
allow_origins=["*"]
```

**Production (Update in `backend/main.py`):**
```python
# Restrict to your Vercel frontend
allow_origins=[
    "https://your-app.vercel.app",
    "https://www.your-domain.com"
]
```

### Backend Features

✅ **Singleton Pattern**: Model loaded once on startup
✅ **Auto Provider Selection**: GPU (CUDA) → CPU fallback
✅ **Input Validation**: Pydantic models with custom validators
✅ **Error Handling**: Comprehensive error messages & logging
✅ **Async Processing**: All endpoints are async
✅ **Base64 Support**: Handles data URLs and plain base64
✅ **Top-K Predictions**: Configurable (1-10)

---

## 🧪 Testing

### Model Testing

```bash
# Test on specific image
python inference.py ./data/stop/1.png

# Test on random samples from dataset
python inference.py
```

**Example Output:**
```
======================================================================
Analyzing: ./data/stop/1.png
======================================================================

Top 5 Predictions:
──────────────────────────────────────────────────────────────────────
1. stop                      95.23% ██████████████████████████████████
2. wait                       2.45% ██
3. understand                 1.12% █
4. problem                    0.67% 
5. careful                    0.53% 
──────────────────────────────────────────────────────────────────────

✓ Predicted Sign: STOP
  Confidence: 95.23%
======================================================================
```

### API Testing

#### Automated Tests
```bash
# Make sure server is running
python backend/main.py

# In another terminal
python backend/test_api.py
```

#### Manual Testing (PowerShell)

**Health Check:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

**Prediction:**
```powershell
# Read and encode image
$imagePath = ".\data\stop\1.png"
$imageBytes = [System.IO.File]::ReadAllBytes($imagePath)
$base64Image = [System.Convert]::ToBase64String($imageBytes)

# Send prediction request
$body = @{
    image = $base64Image
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

#### Interactive Testing

Open http://localhost:8000/docs for Swagger UI where you can:
- Test all endpoints interactively
- See request/response schemas
- Try predictions directly in browser

---

## 🌐 Deployment

### Update CORS for Production

**File:** `backend/main.py` (around line 233)

```python
# Change this:
allow_origins=["*"]

# To this:
allow_origins=["https://your-app.vercel.app"]
```

### Deployment Options

#### 1. Render (Recommended - Free Tier)

1. Push code to GitHub
2. Go to https://render.com → New Web Service
3. Connect repository
4. **Root directory**: `backend`
5. **Build command**: `pip install -r ../requirements.txt`
6. **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Set environment variables:
   - `MODEL_PATH=./model.onnx`
   - `LABELS_PATH=./class_labels.txt`
   - `ALLOWED_ORIGINS=https://your-frontend.vercel.app`
8. Deploy!

#### 2. Railway

1. Push to GitHub
2. Go to https://railway.app → New Project
3. Deploy from GitHub
4. Set root directory to `backend`
5. Railway auto-detects Python and runs

#### 3. Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY backend/ .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t sli-backend .
docker run -p 8000:8000 sli-backend
```

### Environment Variables (Production)

Set these in your deployment platform:
- `PORT`: 8000 (or platform default)
- `ALLOWED_ORIGINS`: Your frontend URL
- `MODEL_PATH`: `./model.onnx`
- `LABELS_PATH`: `./class_labels.txt`

---

## 🔧 Technical Specifications

### Model Details

- **Architecture**: MobileNetV2 + Custom Layers
- **Input**: 224×224×3 RGB images
- **Output**: 44 class probabilities
- **Parameters**: ~3M total
- **Format**: ONNX (Open Neural Network Exchange)
- **Size**: ~9MB

### Training Configuration

```python
IMG_SIZE = 224 × 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 50 (with early stopping)
EPOCHS_STAGE2 = 30 (with early stopping)
LEARNING_RATE_STAGE1 = 0.0001
LEARNING_RATE_STAGE2 = 0.00001
OPTIMIZER = Adam
LOSS = Categorical Cross-Entropy
```

### Dataset Statistics

- **Classes**: 44 sign language phrases
- **Images per class**: ~40
- **Total images**: ~1,760
- **Training set**: ~1,400 (80%)
- **Validation set**: ~360 (20%)

### Performance Metrics

**Model:**
- Validation Accuracy: 85-95%
- Top-3 Accuracy: 95-99%
- Inference Time: <50ms (GPU), <200ms (CPU)

**Backend:**
- Cold Start: ~2-3 seconds (one-time)
- Inference Time: 30-100ms per image
- Memory Usage: ~500MB
- Concurrent Requests: ✅ Supported (async)

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Problem:** GPU/RAM runs out of memory during training

**Solution:**
```python
# Edit train.py, reduce batch size
BATCH_SIZE = 16  # or 8
```

#### 2. TensorFlow Not Using GPU

**Problem:** Training is slow, GPU not detected

**Check:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Solution:** Install CUDA and cuDNN compatible with TensorFlow

#### 3. Model Files Not Found

**Error:** `FileNotFoundError: Model file not found`

**Solution:**
```bash
# Train the model first
python train.py

# Verify files exist
dir backend\model.onnx
dir backend\class_labels.txt
```

#### 4. CORS Errors (Browser)

**Error:** `Access-Control-Allow-Origin` blocked

**Solution:** 
- For development: Already configured for all origins
- For production: Update `backend/main.py` with your frontend URL

#### 5. Import Errors

**Problem:** Module not found errors

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

#### 6. Slow API Inference

**Issue:** Predictions take too long

**Solutions:**
1. Use GPU: `pip install onnxruntime-gpu`
2. Check provider in logs (should show `CUDAExecutionProvider`)
3. Reduce image size before sending

#### 7. Port Already in Use

**Error:** Address already in use

**Solution:**
```python
# Change port in backend/main.py
uvicorn.run("main:app", port=8001)  # Instead of 8000
```

---

## 📊 Performance Optimization

### Speed Up Training

1. **Use GPU**: 10-20× faster than CPU
2. **Increase batch size**: Better GPU utilization
3. **Data pipeline**: Already optimized with `prefetch`

### Improve Accuracy

1. **More data**: Collect 100+ images per class
2. **Balance dataset**: Equal images across classes
3. **Better augmentation**: Experiment with parameters
4. **Hyperparameter tuning**: Try different learning rates

### Reduce Model Size

1. **Quantization**: Convert to INT8
2. **Pruning**: Remove unnecessary weights
3. **Smaller base**: Try MobileNetV3-Small

---

## 📚 Additional Resources

### Documentation
- **TensorFlow**: https://www.tensorflow.org/
- **ONNX**: https://onnx.ai/
- **FastAPI**: https://fastapi.tiangolo.com/
- **MobileNetV2 Paper**: https://arxiv.org/abs/1801.04381

### Tools
- **TensorBoard**: Visualize training progress
- **Netron**: Visualize ONNX models
- **Postman**: Test API endpoints

---

## ✅ Complete Workflow

```
1. Install Dependencies
   pip install -r requirements.txt
        ↓
2. Verify Dataset (Optional)
   python verify_dataset.py
        ↓
3. Train Model
   python train.py
   ├─ Stage 1: Frozen base training
   └─ Stage 2: Fine-tuning
        ↓
4. Test Model
   python inference.py
        ↓
5. Start Backend API
   python backend/main.py
        ↓
6. Test API
   Open http://localhost:8000/docs
        ↓
7. Integrate with React Frontend
   Use /predict endpoint
        ↓
8. Deploy to Production
   Update CORS → Deploy to Render/Railway
```

---

## 🎯 Features Summary

### ✅ Training Pipeline
- MobileNetV2 transfer learning
- Two-stage training (frozen + fine-tuning)
- Data augmentation
- Early stopping
- Learning rate scheduling
- Model checkpointing
- ONNX export
- Training visualization

### ✅ Backend API
- FastAPI application
- 5 REST endpoints
- Base64 image handling
- Async processing
- Error handling & logging
- CORS enabled
- Interactive documentation
- Health checks

### ✅ Testing & Utilities
- Dataset verification
- Model inference testing
- Automated API tests
- Interactive launcher

---

## 📝 Environment Variables

Create `backend/.env` file (optional):

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# CORS - Update for production!
ALLOWED_ORIGINS=*

# Model Configuration
MODEL_PATH=./model.onnx
LABELS_PATH=./class_labels.txt

# Logging
LOG_LEVEL=INFO
```

---

## 🎓 Tips for Better Results

1. **More Training Data**: Collect 50-100+ images per class
2. **Balanced Dataset**: Ensure equal images across all classes
3. **Quality Images**: Clear, well-lit, consistent backgrounds
4. **Diverse Angles**: Capture signs from different angles
5. **Data Augmentation**: Already enabled in training
6. **Fine-tuning**: Experiment with unfreezing more/fewer layers
7. **Learning Rate**: Try different values if accuracy plateaus

---

## 🆘 Getting Help

### Check These First
1. **Interactive API Docs**: http://localhost:8000/docs
2. **Health Endpoint**: http://localhost:8000/health
3. **Logs**: Check console output for errors
4. **Test Scripts**: Run `python inference.py` and `python backend/test_api.py`

### Common Commands
```bash
# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check ONNX Runtime
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Test backend
curl http://localhost:8000/health
```

---

## 🎉 Summary

You now have a **complete, production-ready** sign language recognition system:

✅ **Training Pipeline**: State-of-the-art MobileNetV2 transfer learning
✅ **ONNX Export**: Production-ready model format
✅ **FastAPI Backend**: Complete REST API with 5 endpoints
✅ **Testing Tools**: Comprehensive testing utilities
✅ **Documentation**: Complete guide in single file
✅ **Deployment Ready**: Easy deployment to cloud platforms

**Ready to start?**

```bash
# Quick start
run.bat

# Or manual steps
pip install -r requirements.txt
python train.py
python backend/main.py
```

**Happy Coding!** 🚀💙

---

**Project:** Sign Language Recognition  
**Model:** MobileNetV2 Transfer Learning  
**Backend:** FastAPI + ONNX Runtime  
**Dataset:** 44 sign language phrases
