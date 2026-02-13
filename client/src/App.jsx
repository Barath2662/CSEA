import React, { useState } from 'react'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { Upload, Camera, Settings, ShieldAlert } from 'lucide-react'
import UploadMode from './components/UploadMode'
import LiveMode from './components/LiveMode'

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()

  const [settings, setSettings] = useState({
    speedLimit: 80,
    earThreshold: 0.25,
    skipFrames: 2,
  })

  const navItems = [
    { path: '/', label: 'Upload Video', icon: <Upload size={18} /> },
    { path: '/live', label: 'Live Camera', icon: <Camera size={18} /> },
  ]

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <ShieldAlert size={24} color="#3b82f6" />
          <span>Driver AI Monitor</span>
        </div>

        {navItems.map((item) => (
          <button
            key={item.path}
            className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
            onClick={() => navigate(item.path)}
          >
            {item.icon}
            {item.label}
          </button>
        ))}

        <div style={{ marginTop: '1.5rem' }}>
          <div className="nav-link" style={{ cursor: 'default', gap: '0.5rem' }}>
            <Settings size={18} />
            Settings
          </div>
          <div className="settings-group">
            <div className="setting-row">
              <label>
                Speed Limit (km/h)
                <span className="setting-val">{settings.speedLimit}</span>
              </label>
              <input
                type="range" min={30} max={160} step={5}
                value={settings.speedLimit}
                onChange={(e) => setSettings(s => ({ ...s, speedLimit: +e.target.value }))}
              />
            </div>
            <div className="setting-row">
              <label>
                EAR Threshold
                <span className="setting-val">{settings.earThreshold}</span>
              </label>
              <input
                type="range" min={0.15} max={0.35} step={0.01}
                value={settings.earThreshold}
                onChange={(e) => setSettings(s => ({ ...s, earThreshold: +e.target.value }))}
              />
            </div>
            <div className="setting-row">
              <label>
                Skip Frames
                <span className="setting-val">{settings.skipFrames}</span>
              </label>
              <input
                type="range" min={0} max={10} step={1}
                value={settings.skipFrames}
                onChange={(e) => setSettings(s => ({ ...s, skipFrames: +e.target.value }))}
              />
            </div>
          </div>
        </div>

        <div className="sidebar-footer">
          AI Driver Behavior Monitoring v3.0<br />
          OpenCV · MediaPipe · YOLOv8 · React
        </div>
      </aside>

      {/* Main */}
      <main className="main-content">
        <Routes>
          <Route path="/" element={<UploadMode settings={settings} />} />
          <Route path="/live" element={<LiveMode settings={settings} />} />
        </Routes>
      </main>
    </div>
  )
}
