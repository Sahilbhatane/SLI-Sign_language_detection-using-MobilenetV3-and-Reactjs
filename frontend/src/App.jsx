import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Header from './components/Header';
import Navbar from './components/Navbar';
import WebcamCapture from './components/WebcamCapture';
import DetectionDisplay from './components/DetectionDisplay';
import LanguageSelector from './components/LanguageSelector';
import HistoryTable from './components/HistoryTable';
import { translateText } from './services/translationService';
import { useSignVoice } from './hooks/useSignVoice';
import { DETECTING_PHRASE } from './utils/signSpeech';

function App() {
  const [activeTab, setActiveTab] = useState('detect');
  const [detection, setDetection] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [translation, setTranslation] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [history, setHistory] = useState([]);
  const [backendConnected, setBackendConnected] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [speakTranslation, setSpeakTranslation] = useState(false);

  useSignVoice({
    voiceEnabled,
    speakTranslation,
    selectedLanguage,
    detection,
    translation,
    isTranslating,
  });

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
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleDetection = useCallback(
    (newDetection) => {
      setDetection(newDetection);

      if (!newDetection.phrase || newDetection.phrase === DETECTING_PHRASE) {
        return;
      }

      const historyEntry = {
        ...newDetection,
        id: Date.now(),
        language: selectedLanguage,
        translation: null,
      };

      setHistory((prev) => [...prev, historyEntry]);
    },
    [selectedLanguage]
  );

  useEffect(() => {
    const performTranslation = async () => {
      if (!detection || !detection.phrase) {
        setTranslation('');
        return;
      }

      if (detection.phrase === DETECTING_PHRASE) {
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

  const updateHistoryTranslation = (timestamp, translatedText) => {
    setHistory((prev) =>
      prev.map((item) =>
        item.timestamp === timestamp
          ? { ...item, translation: translatedText, language: selectedLanguage }
          : item
      )
    );
  };

  const handleLanguageChange = (newLanguage) => {
    setSelectedLanguage(newLanguage);
    if (newLanguage === 'en') {
      setSpeakTranslation(false);
    }
  };

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all detection history?')) {
      setHistory([]);
      setDetection(null);
      setTranslation('');
    }
  };

  const tabClass = 'container mx-auto px-6 py-8';

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />

        <main className="flex-1 overflow-y-auto">
          {activeTab === 'home' && (
            <div key="home" className={tabClass}>
              <div className="max-w-4xl mx-auto text-center space-y-8">
                <div className="text-8xl mb-6">👋</div>
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
                      AI-powered recognition of sign language phrases
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

                <button
                  type="button"
                  onClick={() => setActiveTab('detect')}
                  className="mt-8 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-bold text-lg shadow-xl hover:opacity-95 transition-opacity"
                >
                  Start Detecting →
                </button>
              </div>
            </div>
          )}

          {activeTab === 'detect' && (
            <div key="detect" className={`${tabClass} space-y-8`}>
              <WebcamCapture onDetection={handleDetection} isActive={backendConnected} />

              <div className="max-w-4xl mx-auto">
                <DetectionDisplay detection={detection} />
              </div>

              <div className="max-w-4xl mx-auto flex flex-wrap gap-6 items-center bg-gray-800 rounded-xl px-4 py-3 border border-gray-700">
                <label className="flex items-center gap-2 text-gray-200 text-sm cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="rounded border-gray-600"
                    checked={voiceEnabled}
                    onChange={(e) => setVoiceEnabled(e.target.checked)}
                  />
                  Voice on (uses browser speech)
                </label>
                <label
                  className={`flex items-center gap-2 text-sm select-none ${
                    selectedLanguage === 'en'
                      ? 'text-gray-500 cursor-not-allowed'
                      : 'text-gray-200 cursor-pointer'
                  }`}
                >
                  <input
                    type="checkbox"
                    className="rounded border-gray-600"
                    disabled={selectedLanguage === 'en'}
                    checked={speakTranslation}
                    onChange={(e) => setSpeakTranslation(e.target.checked)}
                  />
                  Speak translation
                </label>
              </div>

              <div className="max-w-4xl mx-auto">
                <LanguageSelector
                  selectedLanguage={selectedLanguage}
                  onLanguageChange={handleLanguageChange}
                  translation={translation}
                  isTranslating={isTranslating}
                />
              </div>
            </div>
          )}

          {activeTab === 'history' && (
            <div key="history" className={tabClass}>
              <HistoryTable history={history} onClear={handleClearHistory} />
            </div>
          )}

          {activeTab === 'learn' && (
            <div key="learn" className={tabClass}>
              <div className="max-w-4xl mx-auto bg-gray-800 rounded-xl p-8">
                <h2 className="text-3xl font-bold text-white mb-6">📚 Learn Sign Language</h2>
                <p className="text-gray-300 mb-4">
                  This feature will contain tutorials and guides for learning sign language.
                </p>
                <p className="text-gray-400">Coming soon...</p>
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <div key="settings" className={tabClass}>
              <div className="max-w-4xl mx-auto bg-gray-800 rounded-xl p-8">
                <h2 className="text-3xl font-bold text-white mb-6">⚙️ Settings</h2>
                <div className="space-y-4">
                  <div className="flex items-center justify-between py-3 border-b border-gray-700">
                    <span className="text-gray-300">Backend Status</span>
                    <span
                      className={`px-3 py-1 rounded-full text-sm ${
                        backendConnected
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}
                    >
                      {backendConnected ? '● Connected' : '● Disconnected'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-gray-700">
                    <span className="text-gray-300">Detections in History</span>
                    <span className="text-white font-semibold">{history.length}</span>
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-gray-700">
                    <span className="text-gray-300">Selected Language</span>
                    <span className="text-white font-semibold">{selectedLanguage.toUpperCase()}</span>
                  </div>
                  <div className="flex items-center justify-between py-3">
                    <span className="text-gray-300">Voice output</span>
                    <span className="text-white font-semibold">{voiceEnabled ? 'On' : 'Off'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'about' && (
            <div key="about" className={tabClass}>
              <div className="max-w-4xl mx-auto bg-gray-800 rounded-xl p-8">
                <h2 className="text-3xl font-bold text-white mb-6">ℹ️ About</h2>
                <div className="space-y-4 text-gray-300">
                  <p>
                    <strong className="text-white">Sign Language Interpreter</strong> is an AI-powered
                    application that provides real-time detection and translation of sign language gestures.
                  </p>
                  <h3 className="text-xl font-bold text-white mt-6">Features:</h3>
                  <ul className="list-disc list-inside space-y-2">
                    <li>Real-time sign language detection (MobileNetV3, ONNX)</li>
                    <li>Phrase-level recognition with confidence gating</li>
                    <li>Multi-language translation (LibreTranslate)</li>
                    <li>Optional browser voice output for detections</li>
                    <li>Detection history with confidence scores</li>
                  </ul>
                  <h3 className="text-xl font-bold text-white mt-6">Technologies:</h3>
                  <ul className="list-disc list-inside space-y-2">
                    <li>Frontend: React + Vite + TailwindCSS</li>
                    <li>Backend: FastAPI + ONNX Runtime</li>
                    <li>Model: MobileNetV3-Large (transfer learning)</li>
                    <li>Translation: LibreTranslate API</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
