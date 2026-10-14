# IntelliCourse AI — Phase 1

A Flask + SQLite course builder with a persistent AI-planning conversation, Gemini-backed course generation, course-aware tutoring, saved modules/lessons/quizzes, authentication, and a dark navy/purple responsive UI. Live mode uses Gemini with Google Search grounding when available. Offline Demo Mode is clearly labelled and never claims its sample content is Google-verified.

## Four-phase build plan

1. **Foundation and core flow (this delivery):** app structure, secure registration/login, database models, persistent course-planning chat, Gemini service with deterministic Demo Mode fallback, saved course/module/lesson/assessment records, dashboard, course preview, roadmap, lesson workspace, quizzes, tutor endpoint, planner API, and analytics API.
2. **Learning workflows:** refine roadmap progress and resume behavior, quiz feedback/retries, tutor context/modes, full task editing and course-aware study schedule, and progress-driven recommendations.
3. **Profile and certificates:** achievements, completion rules, PDF certificates with unique IDs, and certificate verification.
4. **Hardening and release:** broader automated tests, ownership/CSRF/security review, responsive and accessibility pass, error states, final documentation, and clean ZIP release.

## Windows setup

Open PowerShell in the extracted `intelli_course_ai` folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window and activate again. Python 3.11 or 3.12 is recommended. Python 3.13 may also work if the installed packages support it.

Open **http://127.0.0.1:5000**. Register an account and open **AI Course Builder**. Demo Mode requires no key; course data is saved in SQLite and survives refreshes/restarts.

## Enable Gemini

1. Create an API key in Google AI Studio: https://aistudio.google.com/apikey
2. Put the key in `.env` (never commit this file):

```dotenv
FLASK_SECRET_KEY=replace-with-a-long-random-secret
DATABASE_URL=sqlite:///intelli_course.db
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
DEMO_MODE=false
```

3. Restart `python app.py`. Course generation requests a complete six-module course with multiple lessons, concrete examples, exercises, quizzes, and reference links. Gemini responses are validated before saving. If Gemini is unavailable or returns an incomplete response, the app uses a clearly labelled offline sample course rather than presenting it as Google-verified. Tutor answers receive the active lesson and full course outline as context. The browser never receives the API key.

## Database

The app calls `db.create_all()` during startup. It creates the SQLite database under Flask's instance directory by default and does not delete or reset it on startup. For a fresh local database, stop the app and back up/remove `instance/intelli_course.db` only if you intentionally want to start over.

## Phase 1 functionality included

- Registration/login/logout, Werkzeug password hashing, Flask-Login, CSRF protection.
- Persistent conversation state and history stored in SQLite.
- Structured Gemini prompt with deterministic demo fallback.
- Validated course structure saved as course, modules, lessons, assessments, and questions.
- Course preview, roadmap, lesson content, tutor panel, backend-scored quizzes.
- Real database-backed analytics API and CRUD study-task API.
- Responsive dark UI and visible Demo Mode/AI Connected status.
- Ownership checks on course, lesson, assessment, and study-task endpoints.

## Current Phase 1 limitations

- The planner supports persistent task creation, completion, and deletion. Rich editing and course-generated scheduling are part of Phase 2.
- Certificates and verification are not issued in Phase 1; they are part of Phase 3. The current certificates API returns an empty list rather than pretending certificates exist.
- Chat preferences are conversational and persistent for the active conversation. Starting a separate course-planning conversation UI is a planned refinement.
- AI output quality depends on the configured provider/model. Always review generated learning material before relying on it.

## Project structure

```text
intelli_course_ai/
  app.py config.py models.py requirements.txt .env.example
  services/ai_service.py services/course_service.py
  templates/ static/css/ static/js/ tests/
```

## Tests

Run:

```powershell
python -m unittest discover -s tests -v
```

The included tests exercise preference collection, the six-module offline course structure, and course validation. For production deployment, set a strong random `FLASK_SECRET_KEY`, use HTTPS, configure secure cookies, and add database migrations and rate limiting.
