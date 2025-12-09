import React, { useState } from "react"
import "./ChatBotPanel.css"
import messageIcon from "../assets/message_icon.png"
import uploadButton from "../assets/upload_button.png"

function ChatBotPanel() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const textareaRef = React.useRef(null)

  const handleSend = () => {
    if (!input.trim()) return
    const newMessage = { sender: "user", text: input }
    setMessages([...messages, newMessage, { sender: "bot", text: "분석 중..." }])
    setInput("")
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-title">
          <img src={messageIcon} alt="chat" />
          <div>
            <p>Assistant</p>
            <h3>ESG AI 챗봇</h3>
          </div>
        </div>
      </div>
      <div className="chat-window">
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
        </div>
        <div className="chat-input">
          <textarea
            ref={textareaRef}
            placeholder="질문을 입력하세요..."
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              if (textareaRef.current) {
                textareaRef.current.style.height = "auto"
                textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
              }
            }}
          />
          <button className="send-btn" onClick={handleSend} disabled={!input.trim()}>
            <img src={uploadButton} alt="send" />
          </button>
        </div>
      </div>
    </div>
  )
}

export default ChatBotPanel
