from datetime import datetime, date
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

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
    modules = db.relationship("Module", backref="course", cascade="all, delete-orphan", order_by="Module.position")
    @property
    def objectives(self):
        import json
        try: return json.loads(self.objectives_json or "[]")
        except Exception: return []

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
        import json
        try: return json.loads(self.objectives_json or "[]")
        except Exception: return []

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
    @property
    def concepts(self):
        import json
        try: return json.loads(self.concepts_json or "[]")
        except Exception: return []
    @property
    def examples(self):
        import json
        try: return json.loads(self.examples_json or "[]")
        except Exception: return []
    @property
    def exercises(self):
        import json
        try: return json.loads(self.exercises_json or "[]")
        except Exception: return []
    @property
    def takeaways(self):
        import json
        try: return json.loads(self.takeaways_json or "[]")
        except Exception: return []

class ChatConversation(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    title = db.Column(db.String(180), default="New course conversation")
    state_json = db.Column(db.Text, default="{}")
    messages = db.relationship("ChatMessage", backref="conversation", cascade="all, delete-orphan", order_by="ChatMessage.created_at")
    @property
    def state(self):
        import json
        try: return json.loads(self.state_json or "{}")
        except Exception: return {}

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
    __table_args__ = (db.UniqueConstraint("user_id", "lesson_id", name="uq_user_lesson_progress"),)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    module_id = db.Column(db.Integer, db.ForeignKey("module.id"), nullable=False, unique=True)
    title = db.Column(db.String(180), default="Module Checkpoint")
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
        import json
        try: return json.loads(self.options_json or "[]")
        except Exception: return []

class AssessmentAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessment.id"), nullable=False)
    score = db.Column(db.Integer, default=0)
    total = db.Column(db.Integer, default=0)
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
    issued_at = db.Column(db.DateTime, default=datetime.utcnow)
