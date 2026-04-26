import React from 'react';

const SettingsModal = ({ isOpen, onClose, llmGrammarEnabled, onLlmGrammarEnabledChange }) => {
  if (!isOpen) return null;

  const stun = import.meta.env.VITE_STUN_URLS || 'stun:stun.l.google.com:19302';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-title"
    >
      <div className="bg-gray-900 border border-gray-700 rounded-2xl max-w-lg w-full shadow-2xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <h2 id="settings-title" className="text-lg font-semibold text-white">
            Settings
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-white text-sm px-2 py-1 rounded-lg hover:bg-gray-800"
          >
            Close
          </button>
        </div>

        <div className="px-6 py-5 space-y-5 text-sm text-gray-300">
          <div className="flex items-center justify-between gap-4">
            <span>LLM grammar on finalize</span>
            <label className="inline-flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                className="rounded border-gray-600"
                checked={llmGrammarEnabled}
                onChange={(e) => onLlmGrammarEnabledChange(e.target.checked)}
              />
              <span className="text-gray-200">{llmGrammarEnabled ? 'On' : 'Off'}</span>
            </label>
          </div>

          <div>
            <p className="text-gray-400 text-xs uppercase mb-1">STUN (from VITE_STUN_URLS)</p>
            <p className="font-mono text-xs text-gray-200 break-all bg-gray-800 rounded-lg p-2 border border-gray-700">
              {stun}
            </p>
          </div>

          <p className="text-xs text-gray-500">
            Stable phrase frames and idle timeout are configured in code (5 frames, 3s). Advanced sliders can
            move here later.
          </p>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
