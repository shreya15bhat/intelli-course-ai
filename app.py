import json, secrets
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, abort, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect, generate_csrf
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
from models import db, User, Course, Module, Lesson, ChatConversation, ChatMessage, LessonProgress, Assessment, Question, AssessmentAttempt, StudyTask
from services.ai_service import AIService
from services.course_service import save_course

app = Flask(__name__); app.config.from_object(Config)
db.init_app(app); csrf=CSRFProtect(app)
login_manager=LoginManager(app); login_manager.login_view='login'
@login_manager.user_loader
def load_user(user_id): return db.session.get(User, int(user_id))

def ai(): return AIService(app.config['GEMINI_API_KEY'], app.config['GEMINI_MODEL'], app.config['DEMO_MODE'])
def owned_course(course_id):
    c=Course.query.filter_by(id=course_id,user_id=current_user.id).first_or_404(); return c
def owned_lesson(lesson_id):
    l=Lesson.query.join(Module).join(Course).filter(Lesson.id==lesson_id,Course.user_id==current_user.id).first_or_404(); return l
def get_conversation():
    cid=session.get('conversation_id')
    conv=ChatConversation.query.filter_by(id=cid,user_id=current_user.id).first() if cid else None
    if not conv:
        conv=ChatConversation(user_id=current_user.id,title='Course planning',state_json='{}'); db.session.add(conv); db.session.commit(); session['conversation_id']=conv.id
        db.session.add(ChatMessage(conversation_id=conv.id,role='assistant',content='Hello! What would you like to learn, and what do you want to achieve?')); db.session.commit()
    return conv

@app.context_processor
def inject_globals(): return {'csrf_token': generate_csrf, 'demo_mode': app.config['DEMO_MODE']}
@app.route('/')
def home():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    return render_template('home.html')
@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        name=request.form.get('name','').strip(); email=request.form.get('email','').strip().lower(); password=request.form.get('password','')
        if len(name)<2 or len(password)<8 or '@' not in email: flash('Enter a name, valid email, and password of at least 8 characters.','error')
        elif User.query.filter_by(email=email).first(): flash('An account with that email already exists.','error')
        else:
            user=User(name=name,email=email,password_hash=generate_password_hash(password)); db.session.add(user); db.session.commit(); login_user(user); return redirect(url_for('dashboard'))
    return render_template('auth.html',mode='register')
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        user=User.query.filter_by(email=request.form.get('email','').strip().lower()).first()
        if user and check_password_hash(user.password_hash,request.form.get('password','')): login_user(user); return redirect(url_for('dashboard'))
        flash('Email or password is incorrect.','error')
    return render_template('auth.html',mode='login')
@app.route('/logout',methods=['POST'])
@login_required
def logout(): logout_user(); session.clear(); return redirect(url_for('home'))
@app.route('/dashboard')
@login_required
def dashboard():
    all_courses=Course.query.filter_by(user_id=current_user.id).order_by(Course.updated_at.desc()).all()
    # Avoid showing repeated copies of the same topic in the small dashboard feed.
    courses=[]; seen=set()
    for course in all_courses:
        key=(course.topic or course.title).strip().casefold()
        if key not in seen:
            seen.add(key); courses.append(course)
        if len(courses)==4: break
    return render_template('dashboard.html',courses=courses)
@app.route('/chat')
@login_required
def chat():
    conv=get_conversation(); return render_template('chat.html',conversation=conv)
@app.route('/my-courses')
@login_required
def my_courses(): return render_template('courses.html',courses=Course.query.filter_by(user_id=current_user.id).order_by(Course.updated_at.desc()).all())
@app.route('/courses/<int:course_id>')
@login_required
def course_preview(course_id): return render_template('course_preview.html',course=owned_course(course_id))
@app.route('/courses/<int:course_id>/roadmap')
@login_required
def roadmap(course_id): return render_template('roadmap.html',course=owned_course(course_id))
@app.route('/lessons/<int:lesson_id>')
@login_required
def lesson_page(lesson_id):
    lesson=owned_lesson(lesson_id); modules=lesson.module.course.modules
    return render_template('lesson.html',lesson=lesson,modules=modules)
@app.route('/assessments/<int:assessment_id>')
@login_required
def assessment_page(assessment_id):
    assessment=Assessment.query.join(Module).join(Course).filter(Assessment.id==assessment_id,Course.user_id==current_user.id).first_or_404()
    return render_template('assessment.html',assessment=assessment)
@app.route('/progress')
@login_required
def progress(): return render_template('analytics.html')
@app.route('/study-planner')
@login_required
def study_planner(): return render_template('study_planner.html')
@app.route('/profile')
@login_required
def profile(): return render_template('profile.html')
@app.route('/settings')
@login_required
def settings(): return render_template('settings.html')
@app.route('/explore')
@login_required
def explore(): return render_template('explore.html')

@app.post('/api/chat/message')
@login_required
def api_chat_message():
    data=request.get_json(silent=True) or {}; message=str(data.get('message','')).strip()
    if not message or len(message)>3000: return jsonify(error='Message must contain 1–3000 characters.'),400
    conv=get_conversation(); state=conv.state
    history=[{'role':m.role,'content':m.content} for m in conv.messages[-8:]]
    db.session.add(ChatMessage(conversation_id=conv.id,role='user',content=message)); result=ai().reply(state,message,history)
    conv.state_json=json.dumps(result.get('state',state)); db.session.add(ChatMessage(conversation_id=conv.id,role='assistant',content=result['reply'])); db.session.commit()
    return jsonify(reply=result['reply'],ready=bool(result.get('ready')),state=result.get('state',{}),demo_mode=app.config['DEMO_MODE'])
@app.get('/api/chat/history')
@login_required
def api_chat_history():
    conv=get_conversation(); return jsonify(messages=[{'role':m.role,'content':m.content,'created_at':m.created_at.isoformat()} for m in conv.messages],state=conv.state)
@app.post('/api/courses/generate')
@login_required
def api_generate_course():
    conv=get_conversation(); prefs=conv.state
    if not all(prefs.get(k) for k in ['topic','skill_level','duration_days','daily_minutes','learning_style']): return jsonify(error='Please complete the course profile in chat first.'),400
    payload,source=ai().generate_course(prefs)
    if not AIService.validate_course(payload): return jsonify(error='The course response could not be validated. Please retry.'),502
    try:
        course=save_course(current_user,prefs,payload)
        return jsonify(course_id=course.id,url=url_for('course_preview',course_id=course.id),title=course.title,source=source,demo_mode=(source=='demo'))
    except Exception:
        db.session.rollback(); app.logger.exception('Course save failed'); return jsonify(error='The course could not be saved. Please retry.'),500
@app.get('/api/courses')
@login_required
def api_courses():
    cs=Course.query.filter_by(user_id=current_user.id).all(); return jsonify(courses=[{'id':c.id,'title':c.title,'topic':c.topic,'duration_days':c.duration_days,'url':url_for('course_preview',course_id=c.id)} for c in cs])
@app.get('/api/courses/<int:course_id>')
@login_required
def api_course_detail(course_id):
    c=owned_course(course_id); return jsonify(id=c.id,title=c.title,description=c.description,objectives=c.objectives,modules=[{'id':m.id,'title':m.title,'description':m.description,'lessons':[{'id':l.id,'title':l.title} for l in m.lessons]} for m in c.modules])
@app.put('/api/courses/<int:course_id>')
@login_required
def api_course_update(course_id):
    c=owned_course(course_id); data=request.get_json(silent=True) or {}
    if 'title' in data:
        title=str(data['title']).strip()
        if not title or len(title)>180: return jsonify(error='Title must be 1–180 characters.'),400
        c.title=title
    if 'description' in data: c.description=str(data['description'])[:5000]
    db.session.commit(); return jsonify(ok=True,title=c.title)
@app.delete('/api/courses/<int:course_id>')
@login_required
def api_course_delete(course_id):
    c=owned_course(course_id); db.session.delete(c); db.session.commit(); return jsonify(ok=True)
@app.post('/api/lessons/<int:lesson_id>/complete')
@login_required
def api_lesson_complete(lesson_id):
    l=owned_lesson(lesson_id); p=LessonProgress.query.filter_by(user_id=current_user.id,lesson_id=l.id).first()
    if not p: p=LessonProgress(user_id=current_user.id,lesson_id=l.id); db.session.add(p)
    p.completed=True; p.completed_at=datetime.utcnow(); db.session.commit(); return jsonify(ok=True)
@app.post('/api/ai/tutor')
@login_required
def api_tutor():
    d=request.get_json(silent=True) or {}; q=str(d.get('question','')).strip(); mode=str(d.get('mode','answer'))[:30]
    if not q or len(q)>2000: return jsonify(error='Question must contain 1–2000 characters.'),400
    lesson=None
    if d.get('lesson_id'):
        l=owned_lesson(int(d['lesson_id']))
        course=l.module.course
        lesson={
            'course_title':course.title, 'course_topic':course.topic,
            'course_description':course.description, 'course_objectives':course.objectives,
            'module_title':l.module.title, 'title':l.title, 'explanation':l.explanation,
            'concepts':l.concepts, 'examples':l.examples, 'exercises':l.exercises,
            'takeaways':l.takeaways,
            'course_outline':[{'module':m.title,'description':m.description,
                'lessons':[{'title':x.title,'explanation':x.explanation} for x in m.lessons]}
                for m in course.modules]
        }
    answer,source=ai().tutor(q,lesson,mode)
    return jsonify(answer=answer,source=source)
@app.post('/api/assessments/<int:assessment_id>/submit')
@login_required
def api_assessment_submit(assessment_id):
    assessment=Assessment.query.join(Module).join(Course).filter(Assessment.id==assessment_id,Course.user_id==current_user.id).first_or_404(); d=request.get_json(silent=True) or {}; answers=d.get('answers',{})
    if not isinstance(answers,dict): return jsonify(error='Invalid answers.'),400
    score=0; details=[]
    for q in assessment.questions:
        try: chosen=int(answers.get(str(q.id),-1))
        except (ValueError,TypeError): chosen=-1
        correct=chosen==q.correct_index; score+=int(correct)
        details.append({'question_id':q.id,'chosen':chosen,'correct_index':q.correct_index,'correct':correct,'explanation':q.explanation})
    attempt=AssessmentAttempt(user_id=current_user.id,assessment_id=assessment.id,score=score,total=len(assessment.questions),answers_json=json.dumps(answers)); db.session.add(attempt); db.session.commit()
    return jsonify(score=score,total=len(assessment.questions),percent=round(score*100/len(assessment.questions)) if assessment.questions else 0,details=details)
@app.get('/api/analytics')
@login_required
def api_analytics():
    courses=Course.query.filter_by(user_id=current_user.id).all(); lessons=[l for c in courses for m in c.modules for l in m.lessons]
    done=LessonProgress.query.filter_by(user_id=current_user.id,completed=True).count(); attempts=AssessmentAttempt.query.filter_by(user_id=current_user.id).all()
    by_module=[]
    for c in courses:
        for m in c.modules:
            total=len(m.lessons); complete=LessonProgress.query.filter(LessonProgress.user_id==current_user.id,LessonProgress.completed.is_(True),LessonProgress.lesson_id.in_([l.id for l in m.lessons] or [-1])).count()
            by_module.append({'title':m.title,'completed':complete,'total':total,'percent':round(complete*100/total) if total else 0})
    return jsonify(course_count=len(courses),lesson_count=len(lessons),completed_lessons=done,completion_percent=round(done*100/len(lessons)) if lessons else 0,quiz_attempts=len(attempts),average_quiz_score=round(sum((a.score/a.total*100) for a in attempts if a.total)/len([a for a in attempts if a.total])) if any(a.total for a in attempts) else 0,module_progress=by_module)
@app.get('/api/study-tasks')
@login_required
def api_tasks_get():
    tasks=StudyTask.query.filter_by(user_id=current_user.id).order_by(StudyTask.due_date).all(); return jsonify(tasks=[{'id':t.id,'title':t.title,'due_date':t.due_date.isoformat(),'minutes':t.minutes,'completed':t.completed,'course_id':t.course_id} for t in tasks])
@app.post('/api/study-tasks')
@login_required
def api_tasks_create():
    d=request.get_json(silent=True) or {}; title=str(d.get('title','')).strip()
    if not title or len(title)>180: return jsonify(error='Task title must be 1–180 characters.'),400
    try: due=datetime.strptime(d.get('due_date',''),'%Y-%m-%d').date() if d.get('due_date') else datetime.utcnow().date()
    except ValueError: return jsonify(error='Use YYYY-MM-DD for due date.'),400
    t=StudyTask(user_id=current_user.id,title=title,due_date=due,minutes=max(5,min(600,int(d.get('minutes',30)))),course_id=d.get('course_id')); db.session.add(t); db.session.commit(); return jsonify(id=t.id,ok=True),201
@app.put('/api/study-tasks/<int:task_id>')
@login_required
def api_tasks_update(task_id):
    t=StudyTask.query.filter_by(id=task_id,user_id=current_user.id).first_or_404(); d=request.get_json(silent=True) or {}
    if 'title' in d: t.title=str(d['title']).strip()[:180]
    if 'completed' in d: t.completed=bool(d['completed'])
    if 'minutes' in d: t.minutes=max(5,min(600,int(d['minutes'])))
    if 'due_date' in d:
        try: t.due_date=datetime.strptime(d['due_date'],'%Y-%m-%d').date()
        except ValueError: return jsonify(error='Use YYYY-MM-DD for due date.'),400
    db.session.commit(); return jsonify(ok=True)
@app.delete('/api/study-tasks/<int:task_id>')
@login_required
def api_tasks_delete(task_id):
    t=StudyTask.query.filter_by(id=task_id,user_id=current_user.id).first_or_404(); db.session.delete(t); db.session.commit(); return jsonify(ok=True)
@app.get('/api/certificates')
@login_required
def api_certificates(): return jsonify(certificates=[])

with app.app_context(): db.create_all()
if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=False)
