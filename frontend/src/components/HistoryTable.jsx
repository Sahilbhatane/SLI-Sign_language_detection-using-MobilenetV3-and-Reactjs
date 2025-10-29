import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const HistoryTable = ({ history, onClear }) => {
  if (!history || history.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-gray-800 rounded-xl p-8 text-center"
      >
        <div className="text-5xl mb-3">📋</div>
        <p className="text-gray-400">No detection history yet</p>
        <p className="text-gray-500 text-sm mt-2">Start detecting to see history here</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-800 rounded-xl overflow-hidden shadow-xl"
    >
      {/* Header */}
      <div className="bg-gray-900 px-6 py-4 flex items-center justify-between border-b border-gray-700">
        <div>
          <h2 className="text-white text-xl font-bold flex items-center">
            <span className="text-2xl mr-2">📊</span>
            Detection History
          </h2>
          <p className="text-gray-400 text-sm mt-1">
            {history.length} detection{history.length !== 1 ? 's' : ''} recorded
          </p>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onClear}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-colors"
        >
          🗑️ Clear History
        </motion.button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-900/50">
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                #
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Time
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Detected Sign
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Confidence
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Translation
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Language
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            <AnimatePresence>
              {history.slice().reverse().slice(0, 10).map((item, index) => {
                const confidenceColor = 
                  item.confidence > 0.8 ? 'text-green-400' :
                  item.confidence > 0.5 ? 'text-yellow-400' : 'text-red-400';
                
                const confidenceBg = 
                  item.confidence > 0.8 ? 'bg-green-400/10' :
                  item.confidence > 0.5 ? 'bg-yellow-400/10' : 'bg-red-400/10';

                return (
                  <motion.tr
                    key={item.id || `${item.timestamp}-${index}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ delay: index * 0.05 }}
                    className="hover:bg-gray-700/50 transition-colors"
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                      {history.length - index}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-white font-semibold capitalize">
                        {item.phrase}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 max-w-[100px] h-2 bg-gray-700 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${item.confidence * 100}%` }}
                            className={`h-full ${confidenceBg} ${confidenceColor}`}
                          />
                        </div>
                        <span className={`text-sm font-semibold ${confidenceColor}`}>
                          {(item.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-300">
                      {item.translation || '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                      {item.language ? item.language.toUpperCase() : 'EN'}
                    </td>
                  </motion.tr>
                );
              })}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {/* Footer Stats */}
      <div className="bg-gray-900/50 px-6 py-4 border-t border-gray-700">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-gray-400 text-xs mb-1">Total Detections</p>
            <p className="text-white text-lg font-bold">{history.length}</p>
          </div>
          <div>
            <p className="text-gray-400 text-xs mb-1">Average Confidence</p>
            <p className="text-white text-lg font-bold">
              {history.length > 0
                ? ((history.reduce((sum, item) => sum + item.confidence, 0) / history.length) * 100).toFixed(0)
                : 0}%
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-xs mb-1">Unique Signs</p>
            <p className="text-white text-lg font-bold">
              {new Set(history.map(item => item.phrase)).size}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default HistoryTable;
