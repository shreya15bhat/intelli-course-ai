import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me-before-deployment")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///intelli_course.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true" or not bool(os.getenv("GEMINI_API_KEY", "").strip())
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
    YOUTUBE_MAX_RESULTS = int(os.getenv("YOUTUBE_MAX_RESULTS", "5"))
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    WTF_CSRF_TIME_LIMIT = 3600
