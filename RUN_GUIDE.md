# How to Run the ESG AI Agent App

## Prerequisites
- Python 3.10+
- Node.js & npm

🛠️ Prerequisites (사전 요구 사항)

Python 3.10+

Node.js & npm (LangGraph 시각화 도구 등을 사용할 경우 필요)

Google Chrome Browser (Selenium 크롤링용, 리눅스 환경은 별도 설치 필요)

🚀 Installation (설치 방법)

1. 프로젝트 클론 및 가상환경 설정

# 프로젝트 클론
```bash
git clone [https://github.com/your-repo/ESG_AIagent.git](https://github.com/your-repo/ESG_AIagent.git)
cd ESG_AIagent
```

# 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows
```

2. 의존성 라이브러리 설치
```bash
pip install -r requirements.txt
```

3. Google Chrome 설치 (WSL/Linux 환경 필수)

Windows나 Mac은 설치된 크롬을 사용하지만, WSL(Ubuntu) 환경에서는 별도 설치가 필요합니다.
```bash
chmod +x install_chrome.sh
./install_chrome.sh
```
## Quick Start (Recommended)
You can start both the backend and frontend with a single script:

```bash
./run_app.sh
```

This will:
1. Activate the Python virtual environment.
2. Start the FastAPI backend on port 8000.
3. Start the React frontend on port 5173.

Access the app at: **http://localhost:5173**

## Manual Start
If you prefer to run them separately:

### Backend
```bash
source venv/bin/activate
python -m backend.main
```

### Frontend
```bash
cd frontend
npm run dev
```
