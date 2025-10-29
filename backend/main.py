"""
FastAPI Backend for Sign Language Recognition
Handles image prediction using ONNX model
"""

import os
import json
import base64
import logging
from collections import deque
from io import BytesIO
from typing import Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
import onnxruntime as ort
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from ensemble_inference import ONNXSignLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model_instance = None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    image: str = Field(..., description="Base64-encoded image data")
    top_k: Optional[int] = Field(default=5, ge=1, le=10, description="Number of top predictions to return")
    
    @validator('image')
    def validate_base64(cls, v):
        """Validate base64 encoding"""
        try:
            # Remove data URL prefix if present
            if ',' in v:
                v = v.split(',', 1)[1]
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError('Invalid base64-encoded image data')


class Prediction(BaseModel):
    """Individual prediction result"""
    rank: int
    class_name: str = Field(..., alias="class")
    confidence: float
    confidence_percent: float


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    success: bool
    prediction: str
    confidence: float
    predictions: list[Prediction]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health endpoint"""
    status: str
    model_loaded: bool
    model_path: str
    num_classes: int
    input_shape: list


## ONNXSignLanguageModel now lives in ensemble_inference.py


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI
    Loads model on startup, cleanup on shutdown
    """
    global model_instance
    
    # Startup
    logger.info("=" * 70)
    logger.info("Starting Sign Language Recognition API")
    logger.info("=" * 70)
    
    try:
        model_instance = ONNXSignLanguageModel()
        logger.info("✓ Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sign Language Recognition API")
    model_instance = None


# Initialize FastAPI app
app = FastAPI(
    title="Sign Language Recognition API",
    description="FastAPI backend for sign language phrase recognition using ONNX model",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
# TODO: Restrict origins to Vercel URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins temporarily
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("✓ CORS enabled for all origins (temporary)")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Sign Language Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "classes": "/classes",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for deployment readiness
    Returns model status and configuration
    """
    try:
        if model_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        model_info = model_instance.get_model_info()
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_path=model_instance.model_path,
            num_classes=model_info['num_classes'],
            input_shape=model_info['input_shape']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sign(request: PredictionRequest):
    """
    Predict sign language phrase from base64-encoded image
    
    Args:
        request: PredictionRequest with base64-encoded image
    
    Returns:
        PredictionResponse with top prediction and confidence scores
    """
    import time
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model_instance is None:
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check server logs."
            )
        
        logger.info(f"Received prediction request (top_k={request.top_k})")
        
        # Get predictions
        predictions = model_instance.predict_from_base64(
            request.image,
            top_k=request.top_k
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Threshold logic: if top confidence < 0.6, return Detecting...
        top = predictions[0]
        if top['confidence'] < 0.6:
            pred_label = "Detecting..."
            pred_conf_percent = 0.0
        else:
            pred_label = top['class']
            pred_conf_percent = top['confidence_percent']

        # Build response
        response = PredictionResponse(
            success=True,
            prediction=pred_label,
            confidence=pred_conf_percent,
            predictions=[Prediction(**pred) for pred in predictions],
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"✓ Prediction: {response.prediction} ({response.confidence:.2f}%) - {processing_time:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/classes", tags=["Model Info"])
async def get_classes():
    """
    Get all available sign language classes
    
    Returns:
        Dictionary with list of classes and count
    """
    try:
        if model_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        return {
            "success": True,
            "classes": model_instance.class_labels,
            "count": model_instance.num_classes
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting classes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/model-info", tags=["Model Info"])
async def get_model_info():
    """
    Get detailed model information
    
    Returns:
        Dictionary with model configuration and metadata
    """
    try:
        if model_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        info = model_instance.get_model_info()
        
        return {
            "success": True,
            "model_info": {
                "num_classes": info['num_classes'],
                "input_shape": info['input_shape'],
                "input_name": info['input_name'],
                "providers": info['providers'],
                "model_path": info['model_path']
            },
            "classes": info['classes']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "success": False,
        "error": "Internal server error",
        "detail": str(exc)
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Starting Sign Language Recognition API Server")
    print("=" * 70)
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
