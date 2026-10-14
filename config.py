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
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true" or not bool(os.getenv("GEMINI_API_KEY", "").strip())
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
