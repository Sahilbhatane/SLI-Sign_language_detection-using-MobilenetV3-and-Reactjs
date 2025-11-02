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

# 2. Train model (30-60 min)
python ML/train_v2.py

# 3. Start backend (Terminal 1)
python backend/main.py

# 4. Start frontend (Terminal 2)
cd frontend && npm run dev
```

**URLs:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Project Structure

```
SLI/
├── backend/              # FastAPI server
│   ├── main.py          # API endpoints
│   ├── model_v2.onnx    # AI model
│   └── class_labels.txt # 43 sign classes
├── frontend/            # React app
│   ├── src/            # Components
│   └── dist/           # Production build
├── ML/                 # Training scripts
│   ├── train_v2.py     # Main training
│   └── inference.py    # Test model
├── data/               # Training images (43 classes)
├── run.bat            # Main launcher
└── requirements.txt   # Python dependencies
```

---

## Features

-  Real-time webcam detection
-  43 sign language phrases
-  Multi-language translation (9 languages)
-  Detection history tracking
-  85-95% accuracy
-  FastAPI backend with REST API
-  React frontend with TailwindCSS

---

## Tech Stack

**Backend:**
- Python, TensorFlow, FastAPI, ONNX Runtime

**Frontend:**
- React, Vite, TailwindCSS, Framer Motion

**Model:**
- EfficientNetB3 (transfer learning)
- Input: 300×300 RGB images
- Output: 43 classes
- Size: ~50MB

---

## Troubleshooting

**Port in use:**
```bash
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

**Model not found:**
```bash
python ML/train_v2.py
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

