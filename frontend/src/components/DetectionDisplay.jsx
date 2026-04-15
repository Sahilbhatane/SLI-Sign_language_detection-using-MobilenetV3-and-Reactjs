import React from 'react';
import { DETECTING_PHRASE } from '../utils/signSpeech';

const DetectionDisplay = ({ detection }) => {
  if (!detection) {
    return (
      <div className="text-center py-12 transition-opacity duration-200">
        <div className="text-6xl mb-4">👋</div>
        <p className="text-gray-400 text-lg">Start detection to see results here</p>
      </div>
    );
  }

  const { phrase, confidence, allPredictions, processingTimeMs } = detection;
  const isDetecting = phrase === DETECTING_PHRASE;

  return (
    <div key={phrase} className="space-y-6">
      <div
        className={`rounded-2xl p-8 text-white shadow-xl transition-colors duration-200 ${
          isDetecting
            ? 'bg-gradient-to-r from-gray-600 to-gray-700'
            : 'bg-gradient-to-r from-blue-600 to-indigo-600'
        }`}
      >
        <div className="text-center">
          <p className="text-sm font-semibold mb-2 opacity-90">
            {isDetecting ? 'STATUS' : 'DETECTED SIGN'}
          </p>
          <h2
            className={`font-bold mb-4 capitalize transition-transform duration-200 ${
              isDetecting ? 'text-4xl' : 'text-6xl'
            }`}
          >
            {isDetecting ? 'Hold steady — looking for a clear sign…' : phrase}
          </h2>

          {!isDetecting && (
            <div className="max-w-md mx-auto">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm opacity-90">Confidence</span>
                <span className="text-lg font-bold">{(confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-white/20 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${Math.min(100, confidence * 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {allPredictions && allPredictions.length > 1 && (
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4 flex items-center">
            <span className="text-xl mr-2">🎯</span>
            Model top picks
          </h3>
          <div className="space-y-3">
            {allPredictions.map((pred, index) => (
              <div
                key={`${pred.class}-${index}`}
                className="flex items-center justify-between bg-gray-700 rounded-lg p-3"
              >
                <span className="text-gray-300 capitalize font-medium">
                  {index + 1}. {pred.class}
                </span>
                <div className="flex items-center space-x-3">
                  <div className="w-32 h-2 bg-gray-600 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-gray-400 text-sm w-12 text-right">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-xl p-4 text-center">
          <p className="text-gray-400 text-sm mb-1">Detection time</p>
          <p className="text-white text-xl font-bold">
            {new Date(detection.timestamp).toLocaleTimeString()}
          </p>
        </div>
        <div className="bg-gray-800 rounded-xl p-4 text-center">
          <p className="text-gray-400 text-sm mb-1">Confidence band</p>
          <p
            className={`text-xl font-bold ${
              isDetecting
                ? 'text-gray-400'
                : confidence > 0.8
                  ? 'text-green-400'
                  : confidence > 0.5
                    ? 'text-yellow-400'
                    : 'text-red-400'
            }`}
          >
            {isDetecting ? '—' : confidence > 0.8 ? 'High' : confidence > 0.5 ? 'Medium' : 'Low'}
          </p>
        </div>
        <div className="bg-gray-800 rounded-xl p-4 text-center">
          <p className="text-gray-400 text-sm mb-1">Alternatives</p>
          <p className="text-white text-xl font-bold">{allPredictions ? allPredictions.length : 0}</p>
        </div>
        {processingTimeMs != null && (
          <div className="bg-gray-800 rounded-xl p-4 text-center">
            <p className="text-gray-400 text-sm mb-1">Server latency</p>
            <p className="text-white text-xl font-bold">{Number(processingTimeMs).toFixed(0)} ms</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DetectionDisplay;
