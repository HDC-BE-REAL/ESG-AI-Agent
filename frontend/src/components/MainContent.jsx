import React, { useState } from "react"
import logo from "/B_clean2.png"
import "./MainContent.css"

function MainContent() {
  const [reports, setReports] = useState([
    {
      id: 1,
      title: "2025 건설사 협력사 ESG 체크리스트",
      items: ["환경 관리 체계 점검", "협력사 내부 인권 보호 지침", "공급망 탄소 배출량 관리"],
    },
  ])
  const [search, setSearch] = useState("")

  React.useEffect(() => {
    const handler = () => {
      setReports([
        {
          id: 99,
          title: "샘플 체크리스트",
          items: [
            "문서 ①: ESG 리스크 인식 / 협력사 교육",
            "문서 ②: 원청 요구사항 요약",
            "문서 ③: 결과물 저장 안내",
          ],
        },
      ])
    }
    window.addEventListener("showSample", handler)
    return () => window.removeEventListener("showSample", handler)
  }, [])

  const handleSave = (reportTitle) => {
    alert(`📄 "${reportTitle}" 보고서를 저장했습니다.`)
  }

  return (
    <div className="main-content">
      <div className="main-header">
        <div className="header-title">
          <img
            src={logo}
            alt="logo"
            onClick={() => window.location.reload()}
          />
          <div>
            <p>LLM Output</p>
            <h2>생성된 보고서 / 체크리스트</h2>
          </div>
        </div>
        <div className="header-actions">
          <div className="search-box">
            <span>🔍</span>
            <input
              type="text"
              placeholder="보고서 검색"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>
      </div>

      <div className="report-list">
        {reports
          .filter((report) => report.title.includes(search))
          .map((report) => (
            <div className="report-box" key={report.id}>
              <div className="report-header">
                <h3>{report.title}</h3>
                <button className="save-btn" onClick={() => handleSave(report.title)}>
                  저장
                </button>
              </div>
              <ul>
                {report.items.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
      </div>
    </div>
  )
}

export default MainContent
