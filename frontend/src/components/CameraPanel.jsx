import React from 'react';
import WebcamCapture from './WebcamCapture';

const CameraPanel = ({ onDetection, isActive, transport, onUserMedia }) => {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-gray-400 px-1">
        <span>Camera</span>
        <span className="rounded-full bg-gray-800 px-2 py-0.5 border border-gray-700">
          transport: <span className="text-gray-200 font-mono">{transport}</span>
        </span>
      </div>
      <WebcamCapture onDetection={onDetection} isActive={isActive} onUserMedia={onUserMedia} />
    </div>
  );
};

export default CameraPanel;
