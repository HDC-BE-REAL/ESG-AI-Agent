import React, { useState, useRef } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import "./ChatBotPanel.css"
import messageIcon from "../assets/message_icon.png"
import uploadButton from "../assets/upload_button.png"

function ChatBotPanel() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const textareaRef = useRef(null)

  const handleSend = async () => {
    if (!input.trim()) return
    const userMessage = { sender: "user", text: input }
    const updatedMessages = [...messages, userMessage, { sender: "bot", text: "" }]
    const botIndex = updatedMessages.length - 1
    setMessages(updatedMessages)
    const currentQuery = input
    setInput("")
    setIsLoading(true)
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }

    try {
      const response = await fetch("http://localhost:8000/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: currentQuery }),
      })
      if (!response.ok || !response.body) {
        throw new Error("stream error")
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder("utf-8")
      let botText = ""
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        const events = chunk.split("\n\n")
        for (const event of events) {
          if (!event.trim()) continue
          const dataLine = event.split("data: ").pop()
          if (!dataLine) continue
          const payload = JSON.parse(dataLine)
          if (payload.token) {
            botText += payload.token
            setMessages((prev) => {
              const copy = [...prev]
              copy[botIndex] = { sender: "bot", text: botText }
              return copy
            })
          }
          if (payload.error) {
            throw new Error(payload.error)
          }
        }
      }
    } catch (err) {
      setMessages((prev) => [...prev, { sender: "bot", text: "Error: 응답을 가져오지 못했습니다." }])
    } finally {
      setIsLoading(false)
    }
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
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={markdownComponents}
              >
                {msg.text || (msg.sender === "bot" && isLoading ? "" : msg.text)}
              </ReactMarkdown>
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
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
          />
          <button className="send-btn" onClick={handleSend} disabled={!input.trim() || isLoading}>
            <img src={uploadButton} alt="send" />
          </button>
        </div>
      </div>
    </div>
  )
}

export default ChatBotPanel
  const markdownComponents = {
    h1: ({ node, ...props }) => (
      <h2 className="markdown-h1" {...props} />
    ),
    h2: ({ node, ...props }) => (
      <h3 className="markdown-h2" {...props} />
    ),
    h3: ({ node, ...props }) => (
      <h4 className="markdown-h3" {...props} />
    ),
    strong: ({ node, ...props }) => (
      <strong className="markdown-strong" {...props} />
    ),
    ul: ({ node, ordered, ...props }) => (
      <ul className="markdown-list" {...props} />
    ),
    ol: ({ node, ordered, ...props }) => (
      <ol className="markdown-list" {...props} />
    ),
    code: ({ node, inline, ...props }) => (
      inline ? <code className="markdown-code" {...props} /> : <code className="markdown-code-block" {...props} />
    ),
    pre: ({ node, ...props }) => (
      <pre className="markdown-pre" {...props} />
    ),
  }
