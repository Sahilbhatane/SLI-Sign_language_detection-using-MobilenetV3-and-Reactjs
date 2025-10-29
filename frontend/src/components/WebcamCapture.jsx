import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const WebcamCapture = ({ onDetection, isActive }) => {
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6); // 60%
  const webcamRef = useRef(null);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState(null);
  const [lastCapture, setLastCapture] = useState(null);
  const [fps, setFps] = useState(0);
  const intervalRef = useRef(null);

  const captureAndPredict = useCallback(async () => {
    if (!webcamRef.current || !isActive) {
      console.log('Skipping capture: webcam or isActive not ready');
      return;
    }

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        console.log('No screenshot captured');
        return;
      }

      setLastCapture(Date.now());
      console.log('📸 Captured image, sending to backend...');
      
      // Remove data URL prefix if present
      const base64Image = imageSrc.replace(/^data:image\/\w+;base64,/, '');

      // Send to backend
      const response = await axios.post('/api/predict', {
        image: base64Image,
        top_k: 3
      }, {
        timeout: 5000
      });

      console.log('✅ Backend response:', response.data);

      if (response.data && response.data.predictions) {
        const topPrediction = response.data.predictions[0];
        console.log(`Top prediction: ${topPrediction.class} (${topPrediction.confidence_percent}%)`);
        
        // ALWAYS send detection, let the display component handle threshold
        onDetection({
          phrase: topPrediction.class,
          confidence: topPrediction.confidence, // Use confidence (0-1) not confidence_percent
          allPredictions: response.data.predictions,
          timestamp: new Date().toISOString()
        });
        setError(null);
      }
    } catch (err) {
      console.error('❌ Prediction error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to detect sign');
    }
  }, [isActive, onDetection, confidenceThreshold]);

  useEffect(() => {
    if (isActive && capturing) {
      // Capture every 2 seconds
      intervalRef.current = setInterval(() => {
        captureAndPredict();
        setFps(prev => prev + 1);
      }, 2000);

      // Reset FPS counter every second
      const fpsInterval = setInterval(() => {
        setFps(0);
      }, 1000);

      return () => {
        clearInterval(intervalRef.current);
        clearInterval(fpsInterval);
      };
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }
  }, [isActive, capturing, captureAndPredict]);

  const toggleCapture = () => {
    setCapturing(!capturing);
    if (!capturing) {
      setError(null);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="relative bg-gray-900 rounded-2xl overflow-hidden shadow-2xl"
      >
        {/* Webcam */}
        <div className="relative aspect-video bg-black">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{
              width: 1280,
              height: 720,
              facingMode: 'user'
            }}
            mirrored={false}
            className="w-full h-full object-cover"
            style={{ transform: 'scaleX(1)' }}
            onUserMediaError={(err) => {
              console.error('Webcam error:', err);
              setError('Failed to access webcam. Please check permissions.');
            }}
          />

          {/* Overlay indicators */}
          <AnimatePresence>
            {capturing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 pointer-events-none"
              >
                {/* Recording indicator */}
                <div className="absolute top-4 left-4 flex items-center space-x-2 bg-red-600 text-white px-3 py-2 rounded-full">
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    className="w-3 h-3 bg-white rounded-full"
                  />
                  <span className="text-sm font-semibold">DETECTING</span>
                </div>

                {/* Frame corners */}
                <div className="absolute inset-0 m-8">
                  <div className="absolute top-0 left-0 w-12 h-12 border-t-4 border-l-4 border-blue-500"></div>
                  <div className="absolute top-0 right-0 w-12 h-12 border-t-4 border-r-4 border-blue-500"></div>
                  <div className="absolute bottom-0 left-0 w-12 h-12 border-b-4 border-l-4 border-blue-500"></div>
                  <div className="absolute bottom-0 right-0 w-12 h-12 border-b-4 border-r-4 border-blue-500"></div>
                </div>

                {/* Center crosshair */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="w-32 h-32 border-2 border-blue-400 rounded-full opacity-30"></div>
                  <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-blue-400 opacity-30"></div>
                  <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-blue-400 opacity-30"></div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error overlay */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="absolute top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg max-w-sm"
              >
                <p className="text-sm">{error}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Controls */}
        <div className="bg-gray-800 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleCapture}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 ${
                capturing
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {capturing ? '⏸ Stop Detection' : '▶ Start Detection'}
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={captureAndPredict}
              disabled={!isActive}
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-xl font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              📸 Capture Frame
            </motion.button>
              {/* Confidence threshold */}
              <div className="flex items-center space-x-2 ml-4">
                <label htmlFor="confidence-threshold" className="text-gray-300 text-sm whitespace-nowrap">
                  Min confidence: {(confidenceThreshold * 100).toFixed(0)}%
                </label>
                <input
                  id="confidence-threshold"
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  value={Math.round(confidenceThreshold * 100)}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value) / 100)}
                  className="w-32 accent-blue-500"
                  title="Minimum confidence to accept a detection"
                />
              </div>
          </div>

          <div className="flex items-center space-x-6 text-sm text-gray-300">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-400' : 'bg-gray-500'}`}></div>
              <span>{isActive ? 'Backend Connected' : 'Backend Offline'}</span>
            </div>
            {lastCapture && (
              <div>
                Last capture: {new Date(lastCapture).toLocaleTimeString()}
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Instructions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-4 bg-gray-800 rounded-xl p-4"
      >
        <h3 className="text-white font-semibold mb-2 flex items-center">
          <span className="text-xl mr-2">💡</span>
          Tips for Best Results
        </h3>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>• Ensure good lighting on your hands</li>
          <li>• Keep hands within the frame markers</li>
          <li>• Hold each sign steady for 2-3 seconds</li>
          <li>• Position yourself centered in the webcam</li>
        </ul>
      </motion.div>
    </div>
  );
};

export default WebcamCapture;
