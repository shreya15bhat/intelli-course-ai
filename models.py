from datetime import datetime, date
import json
import secrets
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text

db = SQLAlchemy()

class TimestampMixin:
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    level = db.Column(db.String(40), default="Beginner")
    courses = db.relationship("Course", backref="owner", cascade="all, delete-orphan")
    conversations = db.relationship("ChatConversation", backref="owner", cascade="all, delete-orphan")

class Course(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    title = db.Column(db.String(180), nullable=False)
    description = db.Column(db.Text, default="")
    topic = db.Column(db.String(180), default="")
    skill_level = db.Column(db.String(40), default="Beginner")
    duration_days = db.Column(db.Integer, default=30)
    daily_minutes = db.Column(db.Integer, default=60)
    learning_style = db.Column(db.String(80), default="Mixed")
    learning_goal = db.Column(db.Text, default="")
    desired_outcome = db.Column(db.Text, default="")
    language = db.Column(db.String(60), default="English")
    objectives_json = db.Column(db.Text, default="[]")
    prerequisites_json = db.Column(db.Text, default="[]")
    capstone_project_json = db.Column(db.Text, default="{}")
    modules = db.relationship("Module", backref="course", cascade="all, delete-orphan", order_by="Module.position")
    
    @property
    def objectives(self):
        try:
            return json.loads(self.objectives_json or "[]")
        except Exception:
            return []

    @property
    def prerequisites(self):
        try:
            return json.loads(self.prerequisites_json or "[]")
        except Exception:
            return []

    @property
    def capstone_project(self):
        try:
            return json.loads(self.capstone_project_json or "{}")
        except Exception:
            return {}

    def get_progress_data(self, user_id):
        all_lessons = [l for m in self.modules for l in m.lessons]
        total_lessons = len(all_lessons)
        if total_lessons == 0:
            return {"completed_count": 0, "total_count": 0, "percent": 0, "completed_ids": set()}
        lesson_ids = [l.id for l in all_lessons]
        completed_records = LessonProgress.query.filter(
            LessonProgress.user_id == user_id,
            LessonProgress.completed.is_(True),
            LessonProgress.lesson_id.in_(lesson_ids)
        ).all()
        completed_ids = {r.lesson_id for r in completed_records}
        completed_count = len(completed_ids)
        percent = round((completed_count / total_lessons) * 100)
        return {
            "completed_count": completed_count,
            "total_count": total_lessons,
            "percent": percent,
            "completed_ids": list(completed_ids)
        }

class Module(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False)
    position = db.Column(db.Integer, default=1)
    title = db.Column(db.String(180), nullable=False)
    description = db.Column(db.Text, default="")
    duration_minutes = db.Column(db.Integer, default=60)
    objectives_json = db.Column(db.Text, default="[]")
    lessons = db.relationship("Lesson", backref="module", cascade="all, delete-orphan", order_by="Lesson.position")
    assessment = db.relationship("Assessment", backref="module", uselist=False, cascade="all, delete-orphan")

    @property
    def objectives(self):
        try:
            return json.loads(self.objectives_json or "[]")
        except Exception:
            return []

class Lesson(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    module_id = db.Column(db.Integer, db.ForeignKey("module.id"), nullable=False)
    position = db.Column(db.Integer, default=1)
    title = db.Column(db.String(180), nullable=False)
    explanation = db.Column(db.Text, default="")
    concepts_json = db.Column(db.Text, default="[]")
    examples_json = db.Column(db.Text, default="[]")
    exercises_json = db.Column(db.Text, default="[]")
    takeaways_json = db.Column(db.Text, default="[]")
    video_json = db.Column(db.Text, default="{}")
    coding_exercise_json = db.Column(db.Text, default="{}")

    @property
    def concepts(self):
        try:
            return json.loads(self.concepts_json or "[]")
        except Exception:
            return []

    @property
    def examples(self):
        try:
            return json.loads(self.examples_json or "[]")
        except Exception:
            return []

    @property
    def exercises(self):
        try:
            return json.loads(self.exercises_json or "[]")
        except Exception:
            return []

    @property
    def takeaways(self):
        try:
            return json.loads(self.takeaways_json or "[]")
        except Exception:
            return []

    @property
    def video(self):
        try:
            return json.loads(self.video_json or "{}")
        except Exception:
            return {}

    @property
    def coding_exercise(self):
        try:
            return json.loads(self.coding_exercise_json or "{}")
        except Exception:
            return {}

    def is_completed(self, user_id):
        p = LessonProgress.query.filter_by(user_id=user_id, lesson_id=self.id).first()
        return bool(p and p.completed)

class ChatConversation(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    title = db.Column(db.String(180), default="New course conversation")
    state_json = db.Column(db.Text, default="{}")
    messages = db.relationship("ChatMessage", backref="conversation", cascade="all, delete-orphan", order_by="ChatMessage.created_at")

    @property
    def state(self):
        try:
            return json.loads(self.state_json or "{}")
        except Exception:
            return {}

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("chat_conversation.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

class LessonProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    lesson_id = db.Column(db.Integer, db.ForeignKey("lesson.id"), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime)
    last_accessed_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("user_id", "lesson_id", name="uq_user_lesson_progress"),)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    module_id = db.Column(db.Integer, db.ForeignKey("module.id"), nullable=False, unique=True)
    title = db.Column(db.String(180), default="Module Checkpoint")
    is_final = db.Column(db.Boolean, default=False)
    passing_score = db.Column(db.Integer, default=70)
    questions = db.relationship("Question", backref="assessment", cascade="all, delete-orphan", order_by="Question.position")

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessment.id"), nullable=False)
    position = db.Column(db.Integer, default=1)
    prompt = db.Column(db.Text, nullable=False)
    options_json = db.Column(db.Text, default="[]")
    correct_index = db.Column(db.Integer, default=0)
    explanation = db.Column(db.Text, default="")

    @property
    def options(self):
        try:
            return json.loads(self.options_json or "[]")
        except Exception:
            return []

class AssessmentAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessment.id"), nullable=False)
    score = db.Column(db.Integer, default=0)
    total = db.Column(db.Integer, default=0)
    percentage = db.Column(db.Integer, default=0)
    passed = db.Column(db.Boolean, default=False)
    answers_json = db.Column(db.Text, default="{}")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StudyTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"))
    title = db.Column(db.String(180), nullable=False)
    due_date = db.Column(db.Date, default=date.today)
    minutes = db.Column(db.Integer, default=30)
    completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Achievement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(60), unique=True, nullable=False)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, default="")
    icon = db.Column(db.String(40), default="fa-star")

class UserAchievement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    achievement_id = db.Column(db.Integer, db.ForeignKey("achievement.id"), nullable=False)
    earned_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("user_id", "achievement_id", name="uq_user_achievement"),)

class Certificate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False)
    certificate_id = db.Column(db.String(40), unique=True, nullable=False)
    title = db.Column(db.String(180), default="Certificate of Completion")
    grade = db.Column(db.String(40), default="Distinction")
    verification_code = db.Column(db.String(64), unique=True)
    issued_at = db.Column(db.DateTime, default=datetime.utcnow)
    course = db.relationship("Course")
    user = db.relationship("User")

class CourseTutorMessage(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False, index=True)
    lesson_id = db.Column(db.Integer, db.ForeignKey("lesson.id"), nullable=True, index=True)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    mode = db.Column(db.String(40), default='answer')
    user = db.relationship("User", backref=db.backref("course_tutor_messages", cascade="all, delete-orphan"))
    course = db.relationship("Course", backref=db.backref("course_tutor_messages", cascade="all, delete-orphan", order_by="CourseTutorMessage.created_at"))
    lesson = db.relationship("Lesson", backref=db.backref("course_tutor_messages", cascade="all, delete-orphan"))

def upgrade_db_schema():
    """Safely and idempotently add any missing columns and tables to existing SQLite database."""
    db.create_all()
    with db.engine.connect() as conn:
        commands = [
            ("course", "prerequisites_json", "ALTER TABLE course ADD COLUMN prerequisites_json TEXT DEFAULT '[]'"),
            ("course", "capstone_project_json", "ALTER TABLE course ADD COLUMN capstone_project_json TEXT DEFAULT '{}'"),
            ("lesson", "video_json", "ALTER TABLE lesson ADD COLUMN video_json TEXT DEFAULT '{}'"),
            ("lesson", "coding_exercise_json", "ALTER TABLE lesson ADD COLUMN coding_exercise_json TEXT DEFAULT '{}'"),
            ("lesson_progress", "last_accessed_at", "ALTER TABLE lesson_progress ADD COLUMN last_accessed_at DATETIME"),
            ("assessment", "is_final", "ALTER TABLE assessment ADD COLUMN is_final BOOLEAN DEFAULT 0"),
            ("assessment", "passing_score", "ALTER TABLE assessment ADD COLUMN passing_score INTEGER DEFAULT 70"),
            ("assessment_attempt", "percentage", "ALTER TABLE assessment_attempt ADD COLUMN percentage INTEGER DEFAULT 0"),
            ("assessment_attempt", "passed", "ALTER TABLE assessment_attempt ADD COLUMN passed BOOLEAN DEFAULT 0"),
            ("certificate", "title", "ALTER TABLE certificate ADD COLUMN title VARCHAR(180) DEFAULT 'Certificate of Completion'"),
            ("certificate", "grade", "ALTER TABLE certificate ADD COLUMN grade VARCHAR(40) DEFAULT 'Distinction'"),
            ("certificate", "verification_code", "ALTER TABLE certificate ADD COLUMN verification_code VARCHAR(64) DEFAULT ''")
        ]
        inspector = inspect(db.engine)
        existing_tables = set(inspector.get_table_names())
        for table, col, alter_sql in commands:
            if table in existing_tables:
                existing_cols = {c["name"] for c in inspector.get_columns(table)}
                if col not in existing_cols:
                    try:
                        conn.execute(text(alter_sql))
                        conn.commit()
                    except Exception:
                        pass

