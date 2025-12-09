import React from "react"
import "./Sidebar.css"
import fileIcon from "../assets/file_icon.png"
import messageIcon from "../assets/message_icon.png"

function Sidebar({ isOpen }) {
  return (
    <div className={`sidebar-wrapper ${isOpen ? "open" : "closed"}`}>
      <div className="sidebar-card">
        <div className="sidebar-heading">
          <img src={messageIcon} alt="guide" />
          <div>
            <p className="subtitle">Guide</p>
            <h3>ESG 챗봇 기록</h3>
          </div>
        </div>
        <button className="chat-guide" onClick={() => window.dispatchEvent(new CustomEvent("showSample"))}>
          ✅ ESG 웹 사용 가이드 (기본 대화)
        </button>
      </div>

      <div className="sidebar-card">
        <div className="sidebar-heading">
          <img src={fileIcon} alt="upload" />
          <div>
            <p className="subtitle">Upload</p>
            <h3>파일 업로드</h3>
          </div>
        </div>
        <div className="upload-box">
          <label className="upload-area">
            <span>파일을 드래그하거나 클릭하여 업로드</span>
            <input type="file" multiple />
          </label>
        </div>
      </div>
    </div>
  )
}

export default Sidebar
