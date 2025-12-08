# How to Run the ESG AI Agent App

## Prerequisites
- Python 3.10+
- Node.js & npm

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
