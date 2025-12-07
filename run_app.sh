#!/bin/bash

# Function to kill processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo "Starting ESG AI Agent Application..."

# 1. Start Backend
echo "Starting Backend (FastAPI)..."
source venv/bin/activate
python -m backend.main &
BACKEND_PID=$!
echo "Backend running with PID $BACKEND_PID"

# Wait a moment for backend to initialize
sleep 2

# 2. Start Frontend
echo "Starting Frontend (React)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend running with PID $FRONTEND_PID"

echo "Application is running!"
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:8000"
echo "Press Ctrl+C to stop."

# Keep script running
wait
