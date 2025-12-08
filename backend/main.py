from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import router as api_router

app = FastAPI(title="ESG AI Agent API")
from fastapi.staticfiles import StaticFiles
import os

# Define DATA_DIR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

# CORS Setup
# allow_origins=["*"] cannot be used with allow_credentials=True
# So we use allow_origin_regex to match any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex="https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to ESG AI Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
