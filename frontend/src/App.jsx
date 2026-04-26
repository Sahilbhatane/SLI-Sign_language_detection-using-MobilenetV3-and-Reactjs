import React, { useCallback, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import Header from './components/Header';
import Navbar from './components/Navbar';
import CameraPanel from './components/CameraPanel';
import DetectionPanel from './components/DetectionPanel';
import SentencePanel from './components/SentencePanel';
import ControlsPanel from './components/ControlsPanel';
import SettingsModal from './components/SettingsModal';
import LanguageSelector from './components/LanguageSelector';
import HistoryTable from './components/HistoryTable';
import { translateText } from './services/translationService';
import { correctSentence as llmCorrectSentence } from './services/llmService';
import { useSentencePipeline } from './hooks/useSentencePipeline';
import { DETECTING_PHRASE } from './utils/signSpeech';

const LS_VOICE = 'sli_voice_enabled';
const LS_TTS_PROVIDER = 'sli_tts_provider';
const LS_LANG = 'sli_selected_language';
const LS_LLM_GRAMMAR = 'sli_llm_grammar';
const LS_OFFLINE = 'sli_offline_mode';

function readBoolLs(key, defaultValue) {
  try {
    const v = localStorage.getItem(key);
    if (v === null) return defaultValue;
    return v === '1' || v === 'true';
  } catch {
    return defaultValue;
  }
}

function readTtsProvider() {
  try {
    const v = localStorage.getItem(LS_TTS_PROVIDER);
    if (v === 'server' || v === 'elevenlabs' || v === 'edge') return v;
    return 'edge';
  } catch {
    return 'edge';
  }
}

function readLanguage() {
  try {
    const v = localStorage.getItem(LS_LANG);
    return v && typeof v === 'string' ? v : 'en';
  } catch {
    return 'en';
  }
}

function App() {
  const [activeTab, setActiveTab] = useState('detect');
  const [detection, setDetection] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState(() => readLanguage());
  const [translation, setTranslation] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [history, setHistory] = useState([]);
  const [backendConnected, setBackendConnected] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(() => readBoolLs(LS_VOICE, false));
  const [speakTranslation, setSpeakTranslation] = useState(false);
  const [ttsProvider, setTtsProvider] = useState(() => readTtsProvider());
  const [offlineMode, setOfflineMode] = useState(() => readBoolLs(LS_OFFLINE, false));
  const [llmGrammarEnabled, setLlmGrammarEnabled] = useState(() => readBoolLs(LS_LLM_GRAMMAR, false));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [transport] = useState('rest');
  const [usingAiFallback, setUsingAiFallback] = useState(false);
  const [speakingActive, setSpeakingActive] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(LS_VOICE, voiceEnabled ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, [voiceEnabled]);

  useEffect(() => {
    try {
      localStorage.setItem(LS_TTS_PROVIDER, ttsProvider);
    } catch {
      /* ignore */
    }
  }, [ttsProvider]);

  useEffect(() => {
    try {
      localStorage.setItem(LS_LANG, selectedLanguage);
    } catch {
      /* ignore */
    }
  }, [selectedLanguage]);

  useEffect(() => {
    try {
      localStorage.setItem(LS_OFFLINE, offlineMode ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, [offlineMode]);

  useEffect(() => {
    try {
      localStorage.setItem(LS_LLM_GRAMMAR, llmGrammarEnabled ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, [llmGrammarEnabled]);

  const correctSentence = useMemo(() => {
    if (!llmGrammarEnabled || offlineMode) return undefined;
    return (text) => llmCorrectSentence(text, 256);
  }, [llmGrammarEnabled, offlineMode]);

  const sentence = useSentencePipeline({
    detection,
    voiceEnabled,
    speakTranslation,
    selectedLanguage,
    translation,
    isTranslating,
    ttsProvider,
    llmGrammarEnabled,
    correctSentence,
    offlineMode,
  });

  useEffect(() => {
    const id = window.setInterval(() => {
      try {
        setSpeakingActive(Boolean(window.speechSynthesis?.speaking));
      } catch {
        setSpeakingActive(false);
      }
    }, 250);
    return () => window.clearInterval(id);
  }, []);

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

  const updateHistoryTranslation = useCallback(
    (timestamp, translatedText) => {
      setHistory((prev) =>
        prev.map((item) =>
          item.timestamp === timestamp ? { ...item, translation: translatedText, language: selectedLanguage } : item
        )
      );
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

      if (offlineMode) {
        setTranslation(detection.phrase);
        updateHistoryTranslation(detection.timestamp, detection.phrase);
        return;
      }

      if (selectedLanguage === 'en') {
        setTranslation(detection.phrase);
        updateHistoryTranslation(detection.timestamp, detection.phrase);
        return;
      }

      setIsTranslating(true);
      try {
        const translatedText = await translateText(detection.phrase, selectedLanguage, 'en');
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
  }, [detection, selectedLanguage, offlineMode, updateHistoryTranslation]);

  useEffect(() => {
    let cancelled = false;

    if (!detection?.phrase || detection.phrase === DETECTING_PHRASE) {
      setUsingAiFallback(false);
      return undefined;
    }
    if (offlineMode) {
      setUsingAiFallback(false);
      return undefined;
    }
    const conf = typeof detection.confidence === 'number' ? detection.confidence : 0;
    if (conf >= 0.95) {
      setUsingAiFallback(false);
      return undefined;
    }

    const timer = window.setTimeout(async () => {
      try {
        const resp = await axios.post(
          '/api/fallback',
          {
            recent_predictions: detection.allPredictions ?? [],
            reason: 'low_confidence',
          },
          { timeout: 20_000 }
        );
        if (cancelled) return;
        setUsingAiFallback(Boolean(resp.data?.used_fallback));
      } catch {
        if (!cancelled) setUsingAiFallback(false);
      }
    }, 400);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [detection, offlineMode]);

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

  const exportTranscript = () => {
    const payload = {
      exportedAt: new Date().toISOString(),
      history,
      sentence: {
        buffer: sentence.buffer,
        raw: sentence.rawSentence,
        translation: sentence.sentenceTranslation,
      },
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sli-transcript-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const tabClass = 'container mx-auto px-6 py-8';

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header onOpenSettings={() => setSettingsOpen(true)} />

        <main className="flex-1 overflow-y-auto">
          {activeTab === 'home' && (
            <div key="home" className={tabClass}>
              <div className="max-w-4xl mx-auto text-center space-y-8">
                <div className="text-8xl mb-6">👋</div>
                <h1 className="text-5xl font-bold text-white mb-4">Welcome to Sign Language Interpreter</h1>
                <p className="text-xl text-gray-300 mb-8">
                  Real-time sign language detection and translation powered by AI
                </p>

                <div className="grid md:grid-cols-3 gap-6 mt-12">
                  <div className="bg-gray-800 rounded-xl p-6">
                    <div className="text-4xl mb-3">🎥</div>
                    <h3 className="text-white font-bold mb-2">Real-time Detection</h3>
                    <p className="text-gray-400 text-sm">AI-powered recognition of sign language phrases</p>
                  </div>
                  <div className="bg-gray-800 rounded-xl p-6">
                    <div className="text-4xl mb-3">🌐</div>
                    <h3 className="text-white font-bold mb-2">Multi-language</h3>
                    <p className="text-gray-400 text-sm">Translate to Hindi, Marathi, Spanish, and more</p>
                  </div>
                  <div className="bg-gray-800 rounded-xl p-6">
                    <div className="text-4xl mb-3">📊</div>
                    <h3 className="text-white font-bold mb-2">Track History</h3>
                    <p className="text-gray-400 text-sm">View all detections with timestamps and translations</p>
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
              {usingAiFallback && (
                <div className="max-w-6xl mx-auto rounded-xl border border-amber-600/40 bg-amber-500/10 px-4 py-2 text-amber-200 text-sm">
                  Using AI fallback…
                </div>
              )}

              <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                <CameraPanel onDetection={handleDetection} isActive={backendConnected} transport={transport} />

                <div className="space-y-6">
                  <DetectionPanel detection={detection} />
                  <SentencePanel
                    rawSentence={sentence.rawSentence}
                    buffer={sentence.buffer}
                    isForming={sentence.isForming}
                    sentenceTranslation={sentence.sentenceTranslation}
                    selectedLanguage={selectedLanguage}
                  />
                  <ControlsPanel
                    voiceEnabled={voiceEnabled}
                    onVoiceEnabledChange={setVoiceEnabled}
                    speakTranslation={speakTranslation}
                    onSpeakTranslationChange={setSpeakTranslation}
                    selectedLanguage={selectedLanguage}
                    ttsProvider={ttsProvider}
                    onTtsProviderChange={setTtsProvider}
                    offlineMode={offlineMode}
                    onOfflineModeChange={setOfflineMode}
                    onExportTranscript={exportTranscript}
                    speakingActive={speakingActive}
                  />
                  <LanguageSelector
                    selectedLanguage={selectedLanguage}
                    onLanguageChange={handleLanguageChange}
                    translation={translation}
                    isTranslating={isTranslating}
                  />
                </div>
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
                        backendConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
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
                    <strong className="text-white">Sign Language Interpreter</strong> is an AI-powered application that
                    provides real-time detection and translation of sign language gestures.
                  </p>
                  <h3 className="text-xl font-bold text-white mt-6">Features:</h3>
                  <ul className="list-disc list-inside space-y-2">
                    <li>Real-time sign language detection (ONNX)</li>
                    <li>Phrase-level recognition with confidence gating</li>
                    <li>Multi-language translation</li>
                    <li>Voice mode with sentence pipeline</li>
                    <li>Detection history with confidence scores</li>
                  </ul>
                  <h3 className="text-xl font-bold text-white mt-6">Technologies:</h3>
                  <ul className="list-disc list-inside space-y-2">
                    <li>Frontend: React + Vite + TailwindCSS</li>
                    <li>Backend: FastAPI + ONNX Runtime</li>
                    <li>Optional WebRTC streaming (when enabled)</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        llmGrammarEnabled={llmGrammarEnabled}
        onLlmGrammarEnabledChange={setLlmGrammarEnabled}
      />
    </div>
  );
}

export default App;
