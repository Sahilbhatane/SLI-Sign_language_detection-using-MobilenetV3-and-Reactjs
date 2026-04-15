import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const WebcamCapture = ({ onDetection, isActive }) => {
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6);
  const webcamRef = useRef(null);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState(null);
  const [lastCapture, setLastCapture] = useState(null);
  const [capturesPerSec, setCapturesPerSec] = useState(0);
  const intervalRef = useRef(null);
  const fpsTickRef = useRef(0);
  const fpsIntervalRef = useRef(null);

  const captureAndPredict = useCallback(async () => {
    if (typeof document !== 'undefined' && document.hidden) {
      return;
    }
    if (!webcamRef.current || !isActive) {
      return;
    }

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        return;
      }

      setLastCapture(Date.now());
      fpsTickRef.current += 1;

      const base64Image = imageSrc.replace(/^data:image\/\w+;base64,/, '');

      const response = await axios.post(
        '/api/predict',
        {
          image: base64Image,
          top_k: 3,
          min_confidence: confidenceThreshold,
        },
        { timeout: 5000 }
      );

      if (response.data && response.data.predictions) {
        const pred = response.data.prediction;
        const confPercent = Number(response.data.confidence) || 0;
        const confidence01 = confPercent / 100;

        onDetection({
          phrase: pred,
          confidence: confidence01,
          allPredictions: response.data.predictions,
          timestamp: new Date().toISOString(),
          processingTimeMs: response.data.processing_time_ms,
          minConfidence: response.data.min_confidence ?? confidenceThreshold,
        });
        setError(null);
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to detect sign');
    }
  }, [isActive, onDetection, confidenceThreshold]);

  useEffect(() => {
    if (!isActive || !capturing) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (fpsIntervalRef.current) clearInterval(fpsIntervalRef.current);
      return undefined;
    }

    intervalRef.current = setInterval(() => {
      captureAndPredict();
    }, 2000);

    fpsIntervalRef.current = setInterval(() => {
      setCapturesPerSec(fpsTickRef.current);
      fpsTickRef.current = 0;
    }, 1000);

    return () => {
      clearInterval(intervalRef.current);
      clearInterval(fpsIntervalRef.current);
    };
  }, [isActive, capturing, captureAndPredict]);

  const toggleCapture = () => {
    setCapturing(!capturing);
    if (!capturing) {
      setError(null);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="relative bg-gray-900 rounded-2xl overflow-hidden shadow-2xl transition-opacity duration-200">
        <div className="relative aspect-video bg-black">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{
              width: 640,
              height: 480,
              facingMode: 'user',
            }}
            mirrored={false}
            className="w-full h-full object-cover"
            style={{ transform: 'scaleX(1)' }}
            onUserMediaError={() => {
              setError('Failed to access webcam. Please check permissions.');
            }}
          />

          {capturing && (
            <div className="absolute inset-0 pointer-events-none transition-opacity duration-200">
              <div className="absolute top-4 left-4 flex items-center space-x-2 bg-red-600 text-white px-3 py-2 rounded-full">
                <span className="w-3 h-3 bg-white rounded-full animate-pulse" />
                <span className="text-sm font-semibold">DETECTING</span>
              </div>
              <div className="absolute inset-0 m-8">
                <div className="absolute top-0 left-0 w-12 h-12 border-t-4 border-l-4 border-blue-500" />
                <div className="absolute top-0 right-0 w-12 h-12 border-t-4 border-r-4 border-blue-500" />
                <div className="absolute bottom-0 left-0 w-12 h-12 border-b-4 border-l-4 border-blue-500" />
                <div className="absolute bottom-0 right-0 w-12 h-12 border-b-4 border-r-4 border-blue-500" />
              </div>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <div className="w-32 h-32 border-2 border-blue-400 rounded-full opacity-30" />
                <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-blue-400 opacity-30" />
                <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-blue-400 opacity-30" />
              </div>
            </div>
          )}

          {error && (
            <div className="absolute top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg max-w-sm transition-opacity duration-200">
              <p className="text-sm">{error}</p>
            </div>
          )}
        </div>

        <div className="bg-gray-800 px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4">
            <button
              type="button"
              onClick={toggleCapture}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 ${
                capturing
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {capturing ? '⏸ Stop Detection' : '▶ Start Detection'}
            </button>

            <button
              type="button"
              onClick={captureAndPredict}
              disabled={!isActive}
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-xl font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              📸 Capture Frame
            </button>

            <div className="flex items-center space-x-2">
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
                title="Sent to API as min_confidence"
              />
            </div>
          </div>

          <div className="flex items-center space-x-6 text-sm text-gray-300">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-400' : 'bg-gray-500'}`} />
              <span>{isActive ? 'Backend Connected' : 'Backend Offline'}</span>
            </div>
            {capturing && (
              <span className="text-gray-400">~{capturesPerSec}/s captures</span>
            )}
            {lastCapture && (
              <span>Last: {new Date(lastCapture).toLocaleTimeString()}</span>
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 bg-gray-800 rounded-xl p-4 transition-opacity duration-200">
        <h3 className="text-white font-semibold mb-2 flex items-center">
          <span className="text-xl mr-2">💡</span>
          Tips for Best Results
        </h3>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>• Ensure good lighting on your hands</li>
          <li>• Keep hands within the frame markers</li>
          <li>• Hold each sign steady for 2–3 seconds</li>
          <li>• Position yourself centered in the webcam</li>
        </ul>
      </div>
    </div>
  );
};

export default WebcamCapture;
