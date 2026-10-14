# CourseMind AI — Intelligent Online Learning Platform

CourseMind AI is an enterprise-grade AI-powered online learning platform inspired by Coursera's course structure and learning experience. It transforms any learning goal or technical topic into a complete, structured specialization with modules, lessons, verified educational YouTube video tutorials, interactive in-browser coding exercises (powered by Monaco Editor and Pyodide WebAssembly), a course-aware Gemini AI tutor, graded module checkpoints, a capstone project, and verified certificates of completion.

---

## 🌟 Key Features

### 1. Topic-Specific, Pedagogically Sound Course Generation
- **Dynamic AI Curriculums**: Uses the official Google Gen AI SDK (`google-genai`) with Google Search grounding to design tailored 6-module specializations.
- **Deep Topic Differentiation**:
  - **Python**: Environment setup, variables, control flow, data structures (lists, dicts, sets), functions, OOP, file handling, unit testing, and an Expense Tracker capstone.
  - **Machine Learning**: Preprocessing, train/test leakage prevention, regression, classification metrics (Precision, Recall, ROC-AUC), Random Forests, K-Means clustering, PCA, and a Customer Churn capstone.
  - **Web Development, Java, SQL, DSA**: Deep customized curricula with real code examples and expected outputs.
- **Deterministic High-Fidelity Fallback**: If `GEMINI_API_KEY` is absent or offline, delivers rich authentic curricula clearly labeled as Demo Mode (never fabricating results or claiming unverified sources).

### 2. YouTube Educational Video Integration
- **YouTube Data API v3**: Searches for embeddable educational videos matching exact lesson keywords (`videoCategoryId=27`, `safeSearch=strict`).
- **Curated Verified Video Registry**: Verified educational tutorials from top channels (freeCodeCamp, Programming with Mosh, StatQuest, Corey Schafer, CS50, Traversy Media) for instant playable playback.
- **Responsive Embed Player**: Playable 16:9 iframe embed directly on the lesson page with an external YouTube link fallback and search fallback link.
- **Quota & Memory Caching**: In-process caching prevents duplicate API queries and respects quota limits.

### 3. Coursera-Style Learning Workspace
- **3-Column Responsive Layout**:
  - **Left**: Collapsible curriculum sidebar with module accordions, lesson status checkmarks, and checkpoint links.
  - **Center**: Main learning area with learning objectives, deep explanations, worked code examples, Monaco Editor with live Python sandbox execution, common errors/debugging advice, guided practice, and previous/next/complete navigation.
  - **Right**: Context-aware CourseMind AI Tutor with prompt chips (*Explain Simply*, *Step-by-Step*, *Debug My Code*, *Give Example*).
- **Interactive Code Editor (Monaco Editor)**:
  - Exact VS Code editor engine with `vs-dark` theme and line numbers.
  - **Pyodide WebAssembly Python Sandbox**: Executes real CPython 3.11 in the browser with zero server security vulnerabilities!
  - Terminal console capturing stdout/stderr, execution time, and automated test cases runner.

### 4. Graded Assessments & Checkpoints
- Multiple-choice checkpoints at the end of every module.
- Immediate scoring and percentage calculation (passing threshold: 70%).
- Detailed pedagogical explanations highlighting why the correct answer is right and why distractors are traps.
- Retake capability and attempt history tracking in the SQLite database.

### 5. Capstone Project & Certificate Verification
- Real-world capstone project guidelines, requirements, starter templates, and deliverables checklist.
- Official **Certificate of Completion** generated only when 100% of lessons are completed and all module quizzes are passed.
- Unique certificate ID (e.g. `CM-A8F2BC41`) and public verification URL (`/certificates/<certificate_id>`).
- Built-in print and save-as-PDF support.

### 6. Smart Dashboard & Recommendation Engine
- **Continue Learning**: Prominently displays the active course, next uncompleted lesson, and visual progress bar.
- **Deduplication Engine**: Fixes duplicate card displays and prevents repeated submissions with button debouncing.
- **Personalized Recommendations**: Suggests complementary specializations based on currently enrolled topics and skill levels.

---

## 🛠 Tech Stack

- **Backend**: Python 3.11, Flask 3.1, Flask-SQLAlchemy 3.1, Flask-Login 0.6, Flask-WTF 1.3 (CSRF Protection), Werkzeug.
- **Database**: SQLite with SQLAlchemy ORM (idempotent schema upgrade migrations).
- **AI Engine**: Google Gemini API via official `google-genai` SDK (`gemini-2.5-flash`).
- **Video API**: YouTube Data API v3 with caching and embed verification.
- **Code Execution**: Monaco Editor 0.45 + Pyodide WebAssembly (isolated client sandbox).
- **Frontend**: Jinja2 Templates, HTML5, Vanilla CSS design tokens (dark navy `#070B1E` & purple `#6546F5` palette), Font Awesome 6.5, Bootstrap 5.3 utilities.

---

## 🚀 Quickstart & Windows PowerShell Setup

Open PowerShell in the `intelli_course_ai` folder:

```powershell
# 1. Activate the Python virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional: If execution is restricted, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 2. Verify dependencies
pip install -r requirements.txt

# 3. Create .env configuration file
Copy-Item .env.example .env

# 4. Start the application
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

- Sign up for a new account (or sign in with existing credentials).
- Use **Generate a New Course** on the dashboard for instant specialization creation.
- Open any lesson to access the embedded YouTube player, Monaco Editor, and AI Tutor.

---

## 🔑 Environment Variables (.env)

Edit `.env` to configure your keys (secrets stay on the backend and are never exposed to the frontend):

```dotenv
FLASK_SECRET_KEY=replace-with-a-long-random-secret-key-for-session-cookies
DATABASE_URL=sqlite:///intelli_course.db

# Google Gemini API Key (Get at https://aistudio.google.com/apikey)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
DEMO_MODE=false

# YouTube Data API v3 Key (Get at https://console.cloud.google.com/apis/credentials)
YOUTUBE_API_KEY=your_youtube_api_key_here
YOUTUBE_MAX_RESULTS=5
```

> **Note on Offline Demo Mode**:
> If `GEMINI_API_KEY` is omitted or `DEMO_MODE=true`, the platform runs in **Offline Demo Mode**. It uses deterministic, deep curricula for Python, Java, Machine Learning, Web Development, SQL, and DSA, paired with verified educational tutorials. It is clearly labeled in the top bar and never claims sample text is Google-verified.

---

## 🧪 Running the Test Suite

Run all unit and integration tests using Python's built-in test runner:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This verifies:
1. Topic preference extraction and chat state planning.
2. Distinct topic-specific course generation (Python vs Machine Learning vs Web Dev vs DSA).
3. Schema validation on modules, lessons, quizzes, coding exercises, and capstone projects.
4. YouTube video retrieval (API search, curated videos, search fallbacks).
5. AI tutor modes (*Explain Simply*, *Debug My Code*, *Step-by-Step*).
6. API endpoint workflows (quick-create, lesson completion, quiz grading, tutor, health check).

---

## 📂 Project Directory Structure

```text
intelli_course_ai/
├── app.py                      # Core Express/Flask application routes and controllers
├── config.py                   # Environment configuration loader
├── models.py                   # SQLAlchemy database models & schema migration helpers
├── requirements.txt            # Python dependencies (google-genai, flask, etc.)
├── .env.example                # Sample environment configuration template
├── services/
│   ├── ai_service.py           # Gemini SDK service & topic-specific curriculum generator
│   ├── course_service.py       # Persistence engine for courses, lessons, and quizzes
│   └── youtube_service.py      # YouTube Data API v3 client, cache & video registry
├── templates/
│   ├── base.html               # Master shell with dark navy & purple branding
│   ├── dashboard.html          # Coursera-style dashboard with quick creator & recommendations
│   ├── course_preview.html     # Specialization overview & syllabus accordion
│   ├── lesson.html             # 3-column workspace with Monaco Editor & AI Tutor
│   ├── assessment.html         # Graded checkpoint interface with answer explanations
│   ├── completion.html         # Capstone project guide & Certificate of Completion
│   ├── certificate_view.html   # Public credential verification page
│   ├── courses.html            # Enrolled courses library with search filtering
│   ├── roadmap.html            # Visual curriculum milestone tracker
│   └── auth.html               # Registration & login views
├── static/
│   ├── css/style.css           # Complete design tokens, Monaco & Coursera styling
│   └── js/                     # Client scripts for chatbot, assessments, and planner
└── tests/
    ├── test_ai_service.py      # Unit tests for curriculum & video integration
    └── test_api_endpoints.py   # Integration tests for REST endpoints
```

---

## 🔒 Security & Best Practices

- **Zero Untrusted Server Execution**: User-submitted code runs inside the browser's WebAssembly sandbox via Pyodide, completely isolating the host server.
- **CSRF Protection**: All POST/PUT/DELETE mutations are validated against Flask-WTF CSRF tokens.
- **Environment Isolation**: Secret API keys remain strictly server-side.
- **Ownership Verification**: Database queries strictly filter records by `user_id == current_user.id` to prevent cross-tenant data tampering.
