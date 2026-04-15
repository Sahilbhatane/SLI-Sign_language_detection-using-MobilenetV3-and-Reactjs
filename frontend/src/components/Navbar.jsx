import React from 'react';

const Navbar = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'home', icon: '🏠', label: 'Home' },
    { id: 'detect', icon: '📹', label: 'Detect' },
    { id: 'history', icon: '📊', label: 'History' },
    { id: 'learn', icon: '📚', label: 'Learn' },
    { id: 'settings', icon: '⚙️', label: 'Settings' },
    { id: 'about', icon: 'ℹ️', label: 'About' },
  ];

  return (
    <nav className="bg-gray-800 text-white w-20 flex flex-col items-center py-6 space-y-6 shadow-xl">
      <div className="mb-4">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center text-xl font-bold cursor-default">
          👋
        </div>
      </div>

      <div className="flex-1 flex flex-col space-y-4">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;

          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setActiveTab(item.id)}
              className={`relative flex flex-col items-center p-3 rounded-xl transition-all duration-200 ${
                isActive
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
              title={item.label}
            >
              <span className="text-2xl">{item.icon}</span>
              <span className="text-[10px] mt-1 font-medium">{item.label}</span>
            </button>
          );
        })}
      </div>

      <div className="mt-auto">
        <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center text-sm font-semibold cursor-default hover:bg-gray-600 transition-colors">
          SG
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
