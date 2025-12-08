import { useState } from 'react'
import FileUploader from './components/FileUploader'
import AgentWorkspace from './components/AgentWorkspace'
import ChatInterface from './components/ChatInterface'

function App() {
  const [uploadedFiles, setUploadedFiles] = useState([])

  const handleFileUpload = (files) => {
    setUploadedFiles(prev => [...prev, ...files])
  }

  return (
    <div className="flex h-screen bg-gray-100 overflow-hidden">
      {/* Left Column: File Management */}
      <div className="w-1/4 bg-white border-r border-gray-200 p-4 flex flex-col">
        <h2 className="text-xl font-bold mb-4 text-gray-800">My Data</h2>
        <FileUploader onUpload={handleFileUpload} files={uploadedFiles} />
      </div>

      {/* Center Column: Agent Workspace */}
      <div className="w-2/4 bg-gray-50 flex flex-col border-r border-gray-200">
        <AgentWorkspace />
      </div>

      {/* Right Column: Chat Interface */}
      <div className="w-1/4 bg-white p-4 flex flex-col">
        <h2 className="text-xl font-bold mb-4 text-gray-800">AI Assistant</h2>
        <ChatInterface />
      </div>
    </div>
  )
}

export default App
