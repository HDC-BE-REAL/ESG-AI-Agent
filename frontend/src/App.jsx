import React, { useState } from "react"
import Sidebar from "./components/Sidebar"
import MainContent from "./components/MainContent"
import ChatBotPanel from "./components/ChatBotPanel"
import "./App.css"

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  return (
    <div className="app-shell">
      <button
        onClick={() => setIsSidebarOpen((prev) => !prev)}
        className="sidebar-toggle"
      >
        {isSidebarOpen ? "◀" : "▶"}
      </button>
      <Sidebar isOpen={isSidebarOpen} />
      <div className={`central-panel ${isSidebarOpen ? "" : "expanded"}`}>
        <MainContent />
      </div>
      <ChatBotPanel />
    </div>
  )
}

export default App
