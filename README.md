# Sign Language Recognition

AI-powered sign language detection using deep learning and React.

---

## Quick Start

**Windows:**
```batch
run.bat
```
Choose options from the menu to install, train, and run.

**Manual Setup:**
```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Train model (30–60 min with GPU; longer on CPU)
python ML/train_mobilenet.py

# 3. Start backend (Terminal 1)
python backend/main.py

# 4. Start frontend (Terminal 2)
cd frontend && npm run dev
```

**URLs:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Frontend tests:**
```bash
cd frontend && npm test
```

**Preprocess normalization (Python):**
```bash
python ML/test_preprocess_norm.py
```

---

## Project Structure

```
SLI/
├── backend/              # FastAPI server
│   ├── main.py          # API endpoints
│   ├── ensemble_inference.py  # ONNX inference + preprocessing
│   ├── model_v2.onnx    # AI model
│   └── class_labels.txt # sign phrase classes
├── frontend/            # React app
│   ├── src/             # Components
│   └── dist/            # Production build
├── ML/                  # Training & evaluation
│   ├── train_mobilenet.py   # Main training (MobileNetV3-Large)
│   ├── evaluate_model.py    # Offline metrics
│   └── inference.py         # Test ONNX on images
├── data/                # Training images (one folder per class)
├── run.bat              # Windows launcher
└── requirements.txt   # Python dependencies
```

---

## Features

- Real-time webcam detection
- Phrase-level sign classes (see `backend/class_labels.txt`)
- Multi-language translation (9 languages)
- Optional browser voice output for stable detections
- Detection history tracking
- FastAPI backend with REST API (`/predict` accepts `min_confidence` 0–1)
- React frontend with TailwindCSS

---

## Tech Stack

**Backend:**
- Python, TensorFlow, FastAPI, ONNX Runtime

**Frontend:**
- React, Vite, TailwindCSS

**Model:**
- MobileNetV3-Large (transfer learning), 224×224 RGB
- Training normalization: `image / 127.5 - 1.0` (matched at ONNX inference)

---

## Troubleshooting

**Port in use:**
```bash
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

**Model not found:**
```bash
python ML/train_mobilenet.py
```

**Dependencies error:**
```bash
pip install -r requirements.txt --force-reinstall
```

---

## Resources

- **API Docs**: http://localhost:8000/docs
- **TensorFlow**: https://www.tensorflow.org
- **FastAPI**: https://fastapi.tiangolo.com

---

made by omkar teams 