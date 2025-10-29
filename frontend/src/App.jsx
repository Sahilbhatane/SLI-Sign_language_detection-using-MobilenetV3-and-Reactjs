import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Header from './components/Header';
import Navbar from './components/Navbar';
import WebcamCapture from './components/WebcamCapture';
import DetectionDisplay from './components/DetectionDisplay';
import LanguageSelector from './components/LanguageSelector';
import HistoryTable from './components/HistoryTable';
import { translateText } from './services/translationService';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [activeTab, setActiveTab] = useState('detect');
  const [detection, setDetection] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [translation, setTranslation] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [history, setHistory] = useState([]);
  const [backendConnected, setBackendConnected] = useState(false);

  // Check backend connectivity
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await axios.get('/api/health', { timeout: 3000 });
        setBackendConnected(response.data.status === 'healthy');
      } catch (error) {
        setBackendConnected(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Handle new detection
  const handleDetection = useCallback((newDetection) => {
    setDetection(newDetection);

    // Add to history with unique ID
    const historyEntry = {
      ...newDetection,
      id: Date.now(),
      language: selectedLanguage,
      translation: null
    };

    setHistory(prev => [...prev, historyEntry]);
  }, [selectedLanguage]);

  // Translate detected phrase when detection or language changes
  useEffect(() => {
    const performTranslation = async () => {
      if (!detection || !detection.phrase) {
        setTranslation('');
        return;
      }

      if (selectedLanguage === 'en') {
        setTranslation(detection.phrase);
        updateHistoryTranslation(detection.timestamp, detection.phrase);
        return;
      }

      setIsTranslating(true);
      try {
        const translatedText = await translateText(
          detection.phrase,
          selectedLanguage,
          'en'
        );
        setTranslation(translatedText);
        updateHistoryTranslation(detection.timestamp, translatedText);
      } catch (error) {
        console.error('Translation error:', error);
        setTranslation(`⚠️ ${error.message}`);
      } finally {
        setIsTranslating(false);
      }
    };

    performTranslation();
  }, [detection, selectedLanguage]);

  // Update translation in history
  const updateHistoryTranslation = (timestamp, translatedText) => {
    setHistory(prev =>
      prev.map(item =>
        item.timestamp === timestamp
          ? { ...item, translation: translatedText, language: selectedLanguage }
          : item
      )
    );
  };

  // Handle language change
  const handleLanguageChange = (newLanguage) => {
    setSelectedLanguage(newLanguage);
  };

  // Clear history
  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all detection history?')) {
      setHistory([]);
      setDetection(null);
      setTranslation('');
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Navbar - Left vertical */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main content area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header - Top horizontal */}
        <Header />

        {/* Content area */}
        <main className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {/* Home Tab */}
            {activeTab === 'home' && (
              <motion.div
                key="home"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="container mx-auto px-6 py-8"
              >
                <div className="max-w-4xl mx-auto text-center space-y-8">
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="text-8xl mb-6"
                  >
                    👋
                  </motion.div>
                  <h1 className="text-5xl font-bold text-white mb-4">
                    Welcome to Sign Language Interpreter
                  </h1>
                  <p className="text-xl text-gray-300 mb-8">
                    Real-time sign language detection and translation powered by AI
                  </p>
                  
                  <div className="grid md:grid-cols-3 gap-6 mt-12">
                    <div className="bg-gray-800 rounded-xl p-6">
                      <div className="text-4xl mb-3">🎥</div>
                      <h3 className="text-white font-bold mb-2">Real-time Detection</h3>
                      <p className="text-gray-400 text-sm">
                        AI-powered recognition of 44+ sign language phrases
                      </p>
                    </div>
                    <div className="bg-gray-800 rounded-xl p-6">
                      <div className="text-4xl mb-3">🌐</div>
                      <h3 className="text-white font-bold mb-2">Multi-language</h3>
                      <p className="text-gray-400 text-sm">
                        Translate to Hindi, Marathi, Spanish, and more
                      </p>
                    </div>
                    <div className="bg-gray-800 rounded-xl p-6">
                      <div className="text-4xl mb-3">📊</div>
                      <h3 className="text-white font-bold mb-2">Track History</h3>
                      <p className="text-gray-400 text-sm">
                        View all detections with timestamps and translations
                      </p>
                    </div>
                  </div>

                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setActiveTab('detect')}
                    className="mt-8 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-bold text-lg shadow-xl"
                  >
                    Start Detecting →
                  </motion.button>
                </div>
              </motion.div>
            )}

            {/* Detect Tab */}
            {activeTab === 'detect' && (
              <motion.div
                key="detect"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="container mx-auto px-6 py-8 space-y-8"
              >
                {/* Webcam Feed */}
                <WebcamCapture
                  onDetection={handleDetection}
                  isActive={backendConnected}
                />

                {/* Detection Display */}
                <div className="max-w-4xl mx-auto">
                  <DetectionDisplay detection={detection} />
                </div>

                {/* Language Selector & Translation */}
                <div className="max-w-4xl mx-auto">
                  <LanguageSelector
                    selectedLanguage={selectedLanguage}
                    onLanguageChange={handleLanguageChange}
                    translation={translation}
                    isTranslating={isTranslating}
                  />
                </div>
              </motion.div>
            )}

            {/* History Tab */}
            {activeTab === 'history' && (
              <motion.div
                key="history"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="container mx-auto px-6 py-8"
              >
                <HistoryTable history={history} onClear={handleClearHistory} />
              </motion.div>
            )}

            {/* Learn Tab */}
            {activeTab === 'learn' && (
              <motion.div
                key="learn"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="container mx-auto px-6 py-8"
              >
                <div className="max-w-4xl mx-auto bg-gray-800 rounded-xl p-8">
                  <h2 className="text-3xl font-bold text-white mb-6">
                    📚 Learn Sign Language
                  </h2>
                  <p className="text-gray-300 mb-4">
                    This feature will contain tutorials and guides for learning sign language.
                  </p>
                  <p className="text-gray-400">Coming soon...</p>
                </div>
              </motion.div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <motion.div
                key="settings"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="container mx-auto px-6 py-8"
              >
                <div className="max-w-4xl mx-auto bg-gray-800 rounded-xl p-8">
                  <h2 className="text-3xl font-bold text-white mb-6">
                    ⚙️ Settings
                  </h2>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between py-3 border-b border-gray-700">
                      <span className="text-gray-300">Backend Status</span>
                      <span className={`px-3 py-1 rounded-full text-sm ${
                        backendConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                        {backendConnected ? '● Connected' : '● Disconnected'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between py-3 border-b border-gray-700">
                      <span className="text-gray-300">Detections in History</span>
                      <span className="text-white font-semibold">{history.length}</span>
                    </div>
                    <div className="flex items-center justify-between py-3">
                      <span className="text-gray-300">Selected Language</span>
                      <span className="text-white font-semibold">{selectedLanguage.toUpperCase()}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* About Tab */}
            {activeTab === 'about' && (
              <motion.div
                key="about"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="container mx-auto px-6 py-8"
              >
                <div className="max-w-4xl mx-auto bg-gray-800 rounded-xl p-8">
                  <h2 className="text-3xl font-bold text-white mb-6">
                    ℹ️ About
                  </h2>
                  <div className="space-y-4 text-gray-300">
                    <p>
                      <strong className="text-white">Sign Language Interpreter</strong> is an AI-powered application
                      that provides real-time detection and translation of sign language gestures.
                    </p>
                    <h3 className="text-xl font-bold text-white mt-6">Features:</h3>
                    <ul className="list-disc list-inside space-y-2">
                      <li>Real-time sign language detection using MobileNetV2</li>
                      <li>Support for 44+ sign language phrases</li>
                      <li>Multi-language translation (Hindi, Marathi, Spanish, etc.)</li>
                      <li>Detection history with confidence scores</li>
                      <li>Modern, responsive UI with smooth animations</li>
                    </ul>
                    <h3 className="text-xl font-bold text-white mt-6">Technologies:</h3>
                    <ul className="list-disc list-inside space-y-2">
                      <li>Frontend: React + Vite + TailwindCSS</li>
                      <li>Backend: FastAPI + ONNX Runtime</li>
                      <li>Model: MobileNetV2 (Transfer Learning)</li>
                      <li>Translation: LibreTranslate API</li>
                      <li>Animations: Framer Motion</li>
                    </ul>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
