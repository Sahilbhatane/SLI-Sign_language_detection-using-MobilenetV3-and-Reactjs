import React from 'react';

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white shadow-lg">
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center space-x-3">
          <div className="text-3xl">👋</div>
          <div>
            <h1 className="text-2xl font-bold">Sign Language Interpreter</h1>
            <p className="text-sm text-blue-100">Real-time detection & translation</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-sm">Live</span>
          </div>
          <div className="text-sm bg-white/10 px-3 py-1 rounded-full">
            {new Date().toLocaleDateString('en-US', {
              weekday: 'short',
              year: 'numeric',
              month: 'short',
              day: 'numeric',
            })}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
