import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const DetectionDisplay = ({ detection }) => {
  if (!detection) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center py-12"
      >
        <div className="text-6xl mb-4">👋</div>
        <p className="text-gray-400 text-lg">
          Start detection to see results here
        </p>
      </motion.div>
    );
  }

  const { phrase, confidence, allPredictions } = detection;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={phrase}
        initial={{ opacity: 0, y: 20, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -20, scale: 0.9 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        className="space-y-6"
      >
        {/* Main Detection */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-8 text-white shadow-xl">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="text-center"
          >
            <p className="text-sm font-semibold mb-2 opacity-90">DETECTED SIGN</p>
            <motion.h2
              className="text-6xl font-bold mb-4 capitalize"
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 0.5 }}
            >
              {phrase}
            </motion.h2>
            
            {/* Confidence bar */}
            <div className="max-w-md mx-auto">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm opacity-90">Confidence</span>
                <span className="text-lg font-bold">{(confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-white/20 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${confidence * 100}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                  className="h-full bg-white rounded-full"
                />
              </div>
            </div>
          </motion.div>
        </div>

        {/* Top Predictions */}
        {allPredictions && allPredictions.length > 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gray-800 rounded-xl p-6"
          >
            <h3 className="text-white font-semibold mb-4 flex items-center">
              <span className="text-xl mr-2">🎯</span>
              Alternative Predictions
            </h3>
            <div className="space-y-3">
              {allPredictions.slice(1).map((pred, index) => (
                <motion.div
                  key={pred.class}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 + index * 0.1 }}
                  className="flex items-center justify-between bg-gray-700 rounded-lg p-3"
                >
                  <span className="text-gray-300 capitalize font-medium">
                    {index + 2}. {pred.class}
                  </span>
                  <div className="flex items-center space-x-3">
                    <div className="w-32 h-2 bg-gray-600 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pred.confidence * 100}%` }}
                        transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                        className="h-full bg-blue-500 rounded-full"
                      />
                    </div>
                    <span className="text-gray-400 text-sm w-12 text-right">
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Detection Stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="grid grid-cols-3 gap-4"
        >
          <div className="bg-gray-800 rounded-xl p-4 text-center">
            <p className="text-gray-400 text-sm mb-1">Detection Time</p>
            <p className="text-white text-xl font-bold">
              {new Date(detection.timestamp).toLocaleTimeString()}
            </p>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 text-center">
            <p className="text-gray-400 text-sm mb-1">Accuracy</p>
            <p className={`text-xl font-bold ${
              confidence > 0.8 ? 'text-green-400' : confidence > 0.5 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {confidence > 0.8 ? 'High' : confidence > 0.5 ? 'Medium' : 'Low'}
            </p>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 text-center">
            <p className="text-gray-400 text-sm mb-1">Alternatives</p>
            <p className="text-white text-xl font-bold">
              {allPredictions ? allPredictions.length : 0}
            </p>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default DetectionDisplay;
