import json
import re
import secrets
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, abort, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect, generate_csrf
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from config import Config
from models import (
    db, User, Course, Module, Lesson, ChatConversation, ChatMessage,
    LessonProgress, Assessment, Question, AssessmentAttempt, StudyTask,
    Certificate, CourseTutorMessage, upgrade_db_schema
)
from services.ai_service import AIService
from services.course_service import save_course
from services.youtube_service import get_lesson_video, get_lesson_videos_list

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
csrf = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def ai():
    return AIService(
        app.config.get('GEMINI_API_KEY', ''),
        app.config.get('GEMINI_MODEL', 'gemini-3-flash-preview'),
        app.config.get('DEMO_MODE', False)
    )

def owned_course(course_id):
    return Course.query.filter_by(id=course_id, user_id=current_user.id).first_or_404()

def owned_lesson(lesson_id):
    return Lesson.query.join(Module).join(Course).filter(Lesson.id == lesson_id, Course.user_id == current_user.id).first_or_404()

def get_conversation():
    cid = session.get('conversation_id')
    conv = ChatConversation.query.filter_by(id=cid, user_id=current_user.id).first() if cid else None
    if not conv:
        conv = ChatConversation(user_id=current_user.id, title='Course planning', state_json='{}')
        db.session.add(conv)
        db.session.commit()
        session['conversation_id'] = conv.id
        db.session.add(ChatMessage(
            conversation_id=conv.id,
            role='assistant',
            content="Welcome to CourseMind AI! What subject or skill would you like to master today? (e.g. Python Basics, Machine Learning, Web Development, Java, SQL)"
        ))
        db.session.commit()
    return conv

def get_recommendations_for_user(user_id):
    existing_courses = Course.query.filter_by(user_id=user_id).all()
    existing_topics = " ".join([c.topic.lower() for c in existing_courses])
    
    catalog = [
        {
            "title": "Python Programming Masterclass",
            "topic": "Python Basics",
            "level": "Beginner",
            "duration": 30,
            "modules_count": 6,
            "description": "Master syntax, variables, data structures (lists, dicts, sets), functions, OOP, and automated testing.",
            "icon": "fa-brands fa-python",
            "color": "gradient-purple"
        },
        {
            "title": "Machine Learning & AI Engineering",
            "topic": "Machine Learning",
            "level": "Intermediate",
            "duration": 45,
            "modules_count": 6,
            "description": "Learn data preprocessing, supervised models (regression & trees), evaluation metrics, and predictive capstone pipelines.",
            "icon": "fa-solid fa-brain",
            "color": "gradient-indigo"
        },
        {
            "title": "Full-Stack Web Development",
            "topic": "Web Development",
            "level": "Beginner",
            "duration": 45,
            "modules_count": 6,
            "description": "Build responsive websites using modern HTML5, CSS Flexbox/Grid, JavaScript DOM, APIs, and backend principles.",
            "icon": "fa-solid fa-code",
            "color": "gradient-cyan"
        },
        {
            "title": "SQL Mastery & Relational Databases",
            "topic": "SQL",
            "level": "Beginner",
            "duration": 21,
            "modules_count": 6,
            "description": "Query databases, write complex joins, aggregates, subqueries, and design normalized production schemas.",
            "icon": "fa-solid fa-database",
            "color": "gradient-blue"
        },
        {
            "title": "Data Structures & Algorithms (DSA)",
            "topic": "Data Structures & Algorithms",
            "level": "Intermediate",
            "duration": 60,
            "modules_count": 6,
            "description": "Master linear and non-linear data structures, searching, sorting, recursion, dynamic programming, and Big-O efficiency.",
            "icon": "fa-solid fa-network-wired",
            "color": "gradient-purple"
        },
        {
            "title": "Java Programming & Object-Oriented Design",
            "topic": "Java Programming",
            "level": "Beginner",
            "duration": 30,
            "modules_count": 6,
            "description": "Core Java syntax, classes, inheritance, polymorphism, interfaces, exceptions, and the collections framework.",
            "icon": "fa-brands fa-java",
            "color": "gradient-indigo"
        }
    ]
    # Filter catalog to prioritize topics the learner hasn't enrolled in
    unseen = [item for item in catalog if item["topic"].lower() not in existing_topics]
    return unseen[:3] if unseen else catalog[:3]

@app.context_processor
def inject_globals():
    return {
        'csrf_token': generate_csrf,
        'demo_mode': app.config.get('DEMO_MODE', False),
        'app_name': 'IntelliCourse AI'
    }

# --- PUBLIC & AUTH ROUTES ---

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if len(name) < 2 or len(password) < 8 or '@' not in email:
            flash('Please enter your full name, a valid email, and a password of at least 8 characters.', 'error')
        elif User.query.filter_by(email=email).first():
            flash('This account already exists. Please log in.', 'error')
        else:
            try:
                user = User(name=name, email=email, password_hash=generate_password_hash(password))
                db.session.add(user)
                db.session.commit()
                login_user(user)
                return redirect(url_for('dashboard'))
            except IntegrityError:
                db.session.rollback()
                flash('This account already exists. Please log in.', 'error')
    return render_template('auth.html', mode='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('No account found. Please create an account.', 'error')
        elif not check_password_hash(user.password_hash, password):
            flash('Incorrect password. Please try again.', 'error')
        else:
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('auth.html', mode='login')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('home'))

# --- DASHBOARD & LEARNING WORKSPACE ---

@app.route('/dashboard')
@login_required
def dashboard():
    all_courses = Course.query.filter_by(user_id=current_user.id).order_by(Course.updated_at.desc()).all()
    
    # Deduplicate strictly by unique course ID and topic
    seen_topics = set()
    unique_courses = []
    for c in all_courses:
        key = (c.topic or c.title).strip().casefold()
        if key not in seen_topics:
            seen_topics.add(key)
            unique_courses.append(c)

    # Calculate continue learning card (last accessed course and next uncompleted lesson)
    continue_data = None
    if all_courses:
        latest_course = all_courses[0]
        progress_info = latest_course.get_progress_data(current_user.id)
        next_lesson = None
        for m in latest_course.modules:
            for l in m.lessons:
                if l.id not in progress_info["completed_ids"]:
                    next_lesson = l
                    break
            if next_lesson:
                break
        
        continue_data = {
            "course": latest_course,
            "next_lesson": next_lesson or (latest_course.modules[0].lessons[0] if latest_course.modules and latest_course.modules[0].lessons else None),
            "percent": progress_info["percent"],
            "completed_count": progress_info["completed_count"],
            "total_count": progress_info["total_count"]
        }

    recommendations = get_recommendations_for_user(current_user.id)

    return render_template(
        'dashboard.html',
        courses=unique_courses,
        continue_data=continue_data,
        recommendations=recommendations
    )

@app.route('/my-courses')
@login_required
def my_courses():
    courses = Course.query.filter_by(user_id=current_user.id).order_by(Course.updated_at.desc()).all()
    return render_template('courses.html', courses=courses)

@app.route('/courses/<int:course_id>')
@login_required
def course_preview(course_id):
    c = owned_course(course_id)
    progress_info = c.get_progress_data(current_user.id)
    first_lesson_id = c.modules[0].lessons[0].id if c.modules and c.modules[0].lessons else None
    return render_template('course_preview.html', course=c, progress=progress_info, first_lesson_id=first_lesson_id)

@app.route('/courses/<int:course_id>/roadmap')
@login_required
def roadmap(course_id):
    c = owned_course(course_id)
    progress_info = c.get_progress_data(current_user.id)
    return render_template('roadmap.html', course=c, progress=progress_info)

@app.route('/lessons/<int:lesson_id>')
@login_required
def lesson_page(lesson_id):
    lesson = owned_lesson(lesson_id)
    course = lesson.module.course
    progress_info = course.get_progress_data(current_user.id)
    
    # Update last accessed timestamp
    prog = LessonProgress.query.filter_by(user_id=current_user.id, lesson_id=lesson.id).first()
    if not prog:
        prog = LessonProgress(user_id=current_user.id, lesson_id=lesson.id)
        db.session.add(prog)
    prog.last_accessed_at = datetime.utcnow()
    db.session.commit()

    # Ensure video is present; if empty or fallback while key configured, fetch and persist
    video = lesson.video
    yt_key = app.config.get('YOUTUBE_API_KEY', '')
    if not video or not video.get('video_id') or (yt_key and video.get('source') != 'youtube_api'):
        video = get_lesson_video(lesson.title, course.topic, yt_key, lesson_index=lesson.position)
        lesson.video_json = json.dumps(video)
        db.session.commit()

    return render_template(
        'lesson.html',
        lesson=lesson,
        modules=course.modules,
        course=course,
        progress=progress_info,
        video=video
    )

@app.route('/assessments/<int:assessment_id>')
@login_required
def assessment_page(assessment_id):
    assessment = Assessment.query.join(Module).join(Course).filter(
        Assessment.id == assessment_id,
        Course.user_id == current_user.id
    ).first_or_404()
    
    # Check if there is a previous attempt
    last_attempt = AssessmentAttempt.query.filter_by(
        user_id=current_user.id,
        assessment_id=assessment.id
    ).order_by(AssessmentAttempt.created_at.desc()).first()

    return render_template('assessment.html', assessment=assessment, last_attempt=last_attempt)

@app.route('/courses/<int:course_id>/completion')
@login_required
def course_completion(course_id):
    c = owned_course(course_id)
    progress_info = c.get_progress_data(current_user.id)
    
    # Check if all module assessments are passed
    assessments = [m.assessment for m in c.modules if m.assessment]
    passed_assessments = 0
    for a in assessments:
        att = AssessmentAttempt.query.filter_by(user_id=current_user.id, assessment_id=a.id, passed=True).first()
        if att:
            passed_assessments += 1
            
    is_eligible = (progress_info["percent"] == 100 and passed_assessments == len(assessments))

    return render_template(
        'completion.html',
        course=c,
        progress=progress_info,
        passed_assessments=passed_assessments,
        total_assessments=len(assessments),
        is_eligible=is_eligible
    )

@app.route('/chat')
@login_required
def chat():
    conv = get_conversation()
    return render_template('chat.html', conversation=conv)

@app.route('/progress')
@login_required
def progress():
    return render_template('analytics.html')

@app.route('/study-planner')
@login_required
def study_planner():
    return render_template('study_planner.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/explore')
@login_required
def explore():
    recommendations = get_recommendations_for_user(current_user.id)
    return render_template('explore.html', recommendations=recommendations)

# --- RESTFUL API ENDPOINTS ---

@app.post('/api/courses/quick-create')
@login_required
def api_quick_create():
    data = request.get_json(silent=True) or {}
    topic = str(data.get('topic', '')).strip()
    if not topic or len(topic) < 2:
        return jsonify(error='Please enter a valid course topic.'), 400

    skill_level = str(data.get('skill_level', 'Beginner')).capitalize()
    duration_days = max(7, min(180, int(data.get('duration_days', 30))))
    daily_minutes = max(15, min(240, int(data.get('daily_minutes', 60))))
    learning_style = str(data.get('learning_style', 'Mixed')).capitalize()

    prefs = {
        'topic': topic,
        'skill_level': skill_level,
        'duration_days': duration_days,
        'daily_minutes': daily_minutes,
        'learning_style': learning_style,
        'language': 'English'
    }

    try:
        payload, source = ai().generate_course(prefs)
        if not AIService.validate_course(payload):
            return jsonify(error='Course generation produced an invalid curriculum. Please retry.'), 502

        course = save_course(current_user, prefs, payload, youtube_api_key=app.config.get('YOUTUBE_API_KEY', ''))
        return jsonify(
            course_id=course.id,
            title=course.title,
            url=url_for('course_preview', course_id=course.id),
            source=source,
            demo_mode=(source == 'demo')
        ), 201
    except Exception as exc:
        db.session.rollback()
        app.logger.exception('Quick course creation failed')
        msg = str(exc)
        if msg.startswith('Live AI course generation failed:'):
            msg = msg.replace('Live AI course generation failed:', '').strip()
        return jsonify(error=msg or 'Course generation failed. Please try again.'), 500

@app.post('/api/courses/generate')
@login_required
def api_generate_course():
    conv = get_conversation()
    prefs = conv.state
    if not all(prefs.get(k) for k in ['topic', 'skill_level', 'duration_days', 'daily_minutes', 'learning_style']):
        return jsonify(error='Please complete your course profile in chat before generating.'), 400
    
    try:
        payload, source = ai().generate_course(prefs)
        if not AIService.validate_course(payload):
            return jsonify(error='The course could not be validated against the required curriculum schema. Please retry.'), 502

        course = save_course(current_user, prefs, payload, youtube_api_key=app.config.get('YOUTUBE_API_KEY', ''))
        return jsonify(
            course_id=course.id,
            url=url_for('course_preview', course_id=course.id),
            title=course.title,
            source=source,
            demo_mode=(source == 'demo')
        )
    except Exception as exc:
        db.session.rollback()
        app.logger.exception('Conversational course creation failed')
        msg = str(exc)
        if msg.startswith('Live AI course generation failed:'):
            msg = msg.replace('Live AI course generation failed:', '').strip()
        return jsonify(error=msg or 'Could not generate course. Please try again.'), 500

@app.post('/api/chat/message')
@login_required
def api_chat_message():
    data = request.get_json(silent=True) or {}
    message = str(data.get('message', '')).strip()
    if not message or len(message) > 3000:
        return jsonify(error='Message must be between 1 and 3000 characters.'), 400
    conv = get_conversation()
    state = conv.state
    history = [{'role': m.role, 'content': m.content} for m in conv.messages[-8:]]
    db.session.add(ChatMessage(conversation_id=conv.id, role='user', content=message))
    result = ai().reply(state, message, history)
    conv.state_json = json.dumps(result.get('state', state))
    db.session.add(ChatMessage(conversation_id=conv.id, role='assistant', content=result['reply']))
    db.session.commit()
    return jsonify(
        reply=result['reply'],
        ready=bool(result.get('ready')),
        state=result.get('state', {}),
        demo_mode=app.config.get('DEMO_MODE', True)
    )

@app.get('/api/chat/history')
@login_required
def api_chat_history():
    conv = get_conversation()
    return jsonify(
        messages=[{'role': m.role, 'content': m.content, 'created_at': m.created_at.isoformat()} for m in conv.messages],
        state=conv.state
    )

@app.post('/api/ai/tutor')
@login_required
def api_tutor():
    d = request.get_json(silent=True) or {}
    q = str(d.get('question', '')).strip()
    mode = str(d.get('mode', 'answer'))[:40]
    code_snippet = str(d.get('code_snippet', ''))[:4000]

    if not q or len(q) > 2000:
        return jsonify(error='Please enter a question between 1 and 2000 characters.', can_retry=False), 400

    lesson_id = d.get('lesson_id')
    course_id = d.get('course_id')

    lesson = None
    course = None

    if lesson_id:
        try:
            lesson = owned_lesson(int(lesson_id))
            course = lesson.module.course
        except Exception:
            return jsonify(error='Lesson not found or access denied.', can_retry=False), 404
    elif course_id:
        try:
            course = owned_course(int(course_id))
            if course.modules and course.modules[0].lessons:
                lesson = course.modules[0].lessons[0]
        except Exception:
            return jsonify(error='Course not found or access denied.', can_retry=False), 404
    else:
        return jsonify(error='Either lesson_id or course_id is required.', can_retry=False), 400

    # 1. Full Course Curriculum Outline
    curriculum_outline = []
    all_lessons = []
    for m in course.modules:
        m_lessons = []
        for les in m.lessons:
            m_lessons.append(les.title)
            all_lessons.append(les)
        curriculum_outline.append({
            'module_id': m.id,
            'title': m.title,
            'lessons': m_lessons
        })

    # 2. Identify Previous and Next Lessons
    prev_lesson = None
    next_lesson = None
    if lesson and lesson in all_lessons:
        curr_idx = all_lessons.index(lesson)
        if curr_idx > 0:
            prev_lesson = {'id': all_lessons[curr_idx - 1].id, 'title': all_lessons[curr_idx - 1].title}
        if curr_idx < len(all_lessons) - 1:
            next_lesson = {'id': all_lessons[curr_idx + 1].id, 'title': all_lessons[curr_idx + 1].title}

    # 3. Curriculum Search / Cross-Module Retrieval
    q_lower = q.lower()
    stopwords = {'what', 'when', 'where', 'which', 'who', 'whom', 'this', 'that', 'with', 'from', 'have', 'more', 'some', 'about', 'could', 'would', 'should', 'explain', 'tell', 'give', 'show', 'please'}
    q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', q_lower)) - stopwords
    retrieved_matches = []
    for other_l in all_lessons:
        if lesson and other_l.id == lesson.id:
            continue
        combined_text = f"{other_l.title} {' '.join(other_l.concepts)} {other_l.explanation[:400]}".lower()
        matches = sum(1 for w in q_words if w in combined_text)
        if matches > 0:
            retrieved_matches.append({
                'lesson_id': other_l.id,
                'module_title': other_l.module.title,
                'title': other_l.title,
                'concepts': other_l.concepts,
                'explanation': other_l.explanation[:500],
                'score': matches
            })
    retrieved_matches.sort(key=lambda x: x['score'], reverse=True)
    retrieved_matches = retrieved_matches[:3]

    # 4. Assemble Course Context
    starter_code = ''
    if lesson and lesson.coding_exercise:
        starter_code = lesson.coding_exercise.get('starter_code', '')

    lesson_ctx = {
        'course_id': course.id,
        'course_title': course.title,
        'course_topic': course.topic,
        'skill_level': course.skill_level,
        'course_objectives': course.objectives,
        'curriculum_outline': curriculum_outline,
        'module_title': lesson.module.title if lesson else '',
        'title': lesson.title if lesson else '',
        'explanation': lesson.explanation if lesson else '',
        'concepts': lesson.concepts if lesson else [],
        'examples': lesson.examples if lesson else [],
        'exercises': lesson.exercises if lesson else [],
        'takeaways': lesson.takeaways if lesson else [],
        'starter_code': starter_code,
        'previous_lesson': prev_lesson,
        'next_lesson': next_lesson,
        'retrieved_curriculum_matches': retrieved_matches
    }

    # 5. Retrieve Course-Isolated Conversation History
    past_messages = CourseTutorMessage.query.filter_by(
        user_id=current_user.id,
        course_id=course.id
    ).order_by(CourseTutorMessage.created_at.asc()).all()
    history = [{'role': m.role, 'content': m.content} for m in past_messages[-10:]]

    # 6. Execute Real Gemini Tutor Request
    try:
        answer, source = ai().tutor(q, lesson_ctx, mode, code_snippet, conversation_history=history)
    except Exception as exc:
        app.logger.exception('CourseMind AI Tutor request failed')
        return jsonify(
            error=f"CourseMind AI Tutor error: {str(exc)}",
            can_retry=True,
            question=q,
            mode=mode
        ), 500

    # 7. Persist Conversation Turns
    try:
        user_turn = CourseTutorMessage(
            user_id=current_user.id,
            course_id=course.id,
            lesson_id=lesson.id if lesson else None,
            role='user',
            content=q,
            mode=mode
        )
        assistant_turn = CourseTutorMessage(
            user_id=current_user.id,
            course_id=course.id,
            lesson_id=lesson.id if lesson else None,
            role='assistant',
            content=answer,
            mode=mode
        )
        db.session.add(user_turn)
        db.session.add(assistant_turn)
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify(
        ok=True,
        answer=answer,
        source=source,
        course_id=course.id,
        lesson_id=lesson.id if lesson else None
    )

@app.get('/api/lessons/<int:lesson_id>/tutor/history')
@login_required
def api_lesson_tutor_history(lesson_id):
    l = owned_lesson(lesson_id)
    course = l.module.course
    past_messages = CourseTutorMessage.query.filter_by(
        user_id=current_user.id,
        course_id=course.id
    ).order_by(CourseTutorMessage.created_at.asc()).all()
    return jsonify(
        course_id=course.id,
        lesson_id=l.id,
        messages=[
            {
                'id': m.id,
                'role': m.role,
                'content': m.content,
                'mode': m.mode,
                'lesson_id': m.lesson_id,
                'created_at': m.created_at.isoformat()
            }
            for m in past_messages
        ]
    )

@app.post('/api/courses/<int:course_id>/tutor/clear')
@login_required
def api_course_tutor_clear(course_id):
    c = owned_course(course_id)
    CourseTutorMessage.query.filter_by(
        user_id=current_user.id,
        course_id=c.id
    ).delete()
    db.session.commit()
    return jsonify(ok=True)

@app.get('/api/lessons/<int:lesson_id>/videos')
@login_required
def api_lesson_videos(lesson_id):
    l = owned_lesson(lesson_id)
    video_data = get_lesson_videos_list(
        l.title,
        l.module.course.topic,
        app.config.get('YOUTUBE_API_KEY', ''),
        max_results=app.config.get('YOUTUBE_MAX_RESULTS', 5)
    )
    return jsonify(video_data)

@app.post('/api/lessons/<int:lesson_id>/complete')
@login_required
def api_lesson_complete(lesson_id):
    lesson = owned_lesson(lesson_id)
    prog = LessonProgress.query.filter_by(user_id=current_user.id, lesson_id=lesson.id).first()
    if not prog:
        prog = LessonProgress(user_id=current_user.id, lesson_id=lesson.id)
        db.session.add(prog)
    prog.completed = True
    prog.completed_at = datetime.utcnow()
    prog.last_accessed_at = datetime.utcnow()
    db.session.commit()

    course_progress = lesson.module.course.get_progress_data(current_user.id)
    return jsonify(
        ok=True,
        lesson_id=lesson.id,
        completed=True,
        progress=course_progress
    )

@app.post('/api/assessments/<int:assessment_id>/submit')
@login_required
def api_assessment_submit(assessment_id):
    assessment = Assessment.query.join(Module).join(Course).filter(
        Assessment.id == assessment_id,
        Course.user_id == current_user.id
    ).first_or_404()
    
    d = request.get_json(silent=True) or {}
    answers = d.get('answers', {})
    if not isinstance(answers, dict):
        return jsonify(error='Invalid answers payload.'), 400

    score = 0
    total = len(assessment.questions)
    details = []

    for q in assessment.questions:
        try:
            chosen = int(answers.get(str(q.id), -1))
        except (ValueError, TypeError):
            chosen = -1
        correct = (chosen == q.correct_index)
        if correct:
            score += 1
        details.append({
            'question_id': q.id,
            'chosen': chosen,
            'correct_index': q.correct_index,
            'correct': correct,
            'explanation': q.explanation
        })

    percent = round((score / total) * 100) if total > 0 else 0
    passed = (percent >= assessment.passing_score)

    attempt = AssessmentAttempt(
        user_id=current_user.id,
        assessment_id=assessment.id,
        score=score,
        total=total,
        percentage=percent,
        passed=passed,
        answers_json=json.dumps(answers)
    )
    db.session.add(attempt)
    db.session.commit()

    return jsonify(
        score=score,
        total=total,
        percentage=percent,
        passed=passed,
        passing_score=assessment.passing_score,
        details=details
    )



@app.get('/api/analytics')
@login_required
def api_analytics():
    courses = Course.query.filter_by(user_id=current_user.id).all()
    all_lessons = [l for c in courses for m in c.modules for l in m.lessons]
    completed_records = LessonProgress.query.filter_by(user_id=current_user.id, completed=True).all()
    done_count = len(completed_records)
    attempts = AssessmentAttempt.query.filter_by(user_id=current_user.id).all()
    
    module_progress = []
    for c in courses:
        for m in c.modules:
            total_m = len(m.lessons)
            done_m = LessonProgress.query.filter(
                LessonProgress.user_id == current_user.id,
                LessonProgress.completed.is_(True),
                LessonProgress.lesson_id.in_([l.id for l in m.lessons] or [-1])
            ).count()
            module_progress.append({
                'title': m.title,
                'completed': done_m,
                'total': total_m,
                'percent': round((done_m / total_m) * 100) if total_m else 0
            })

    avg_score = round(sum(a.percentage for a in attempts) / len(attempts)) if attempts else 0

    return jsonify(
        course_count=len(courses),
        lesson_count=len(all_lessons),
        completed_lessons=done_count,
        completion_percent=round((done_count / len(all_lessons)) * 100) if all_lessons else 0,
        quiz_attempts=len(attempts),
        average_quiz_score=avg_score,
        module_progress=module_progress
    )

@app.get('/api/study-tasks')
@login_required
def api_tasks_get():
    tasks = StudyTask.query.filter_by(user_id=current_user.id).order_by(StudyTask.due_date).all()
    return jsonify(tasks=[{
        'id': t.id,
        'title': t.title,
        'due_date': t.due_date.isoformat(),
        'minutes': t.minutes,
        'completed': t.completed,
        'course_id': t.course_id
    } for t in tasks])

@app.post('/api/study-tasks')
@login_required
def api_tasks_create():
    d = request.get_json(silent=True) or {}
    title = str(d.get('title', '')).strip()
    if not title or len(title) > 180:
        return jsonify(error='Task title must be 1–180 characters.'), 400
    try:
        due = datetime.strptime(d.get('due_date', ''), '%Y-%m-%d').date() if d.get('due_date') else datetime.utcnow().date()
    except ValueError:
        return jsonify(error='Use YYYY-MM-DD for due date.'), 400
    t = StudyTask(
        user_id=current_user.id,
        title=title,
        due_date=due,
        minutes=max(5, min(600, int(d.get('minutes', 30)))),
        course_id=d.get('course_id')
    )
    db.session.add(t)
    db.session.commit()
    return jsonify(id=t.id, ok=True), 201

@app.put('/api/study-tasks/<int:task_id>')
@login_required
def api_tasks_update(task_id):
    t = StudyTask.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    d = request.get_json(silent=True) or {}
    if 'title' in d:
        t.title = str(d['title']).strip()[:180]
    if 'completed' in d:
        t.completed = bool(d['completed'])
    if 'minutes' in d:
        t.minutes = max(5, min(600, int(d['minutes'])))
    if 'due_date' in d:
        try:
            t.due_date = datetime.strptime(d['due_date'], '%Y-%m-%d').date()
        except ValueError:
            return jsonify(error='Use YYYY-MM-DD for due date.'), 400
    db.session.commit()
    return jsonify(ok=True)

@app.delete('/api/study-tasks/<int:task_id>')
@login_required
def api_tasks_delete(task_id):
    t = StudyTask.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    db.session.delete(t)
    db.session.commit()
    return jsonify(ok=True)

@app.delete('/api/courses/<int:course_id>')
@login_required
def api_course_delete(course_id):
    c = owned_course(course_id)
    db.session.delete(c)
    db.session.commit()
    return jsonify(ok=True)


@app.get('/api/health')
def api_health():
    return jsonify(
        status='healthy',
        app_name=app.config.get('APP_NAME', 'IntelliCourse AI'),
        demo_mode=app.config.get('DEMO_MODE', False),
        gemini_configured=bool(app.config.get('GEMINI_API_KEY')),
        gemini_model=app.config.get('GEMINI_MODEL', 'gemini-3.1-flash-lite'),
        youtube_configured=bool(app.config.get('YOUTUBE_API_KEY'))
    )

with app.app_context():
    db.create_all()
    upgrade_db_schema()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
