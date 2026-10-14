import unittest
import json
from app import app, db
from models import User, Course, Module, Lesson, LessonProgress, Assessment, Question, AssessmentAttempt, Certificate
from werkzeug.security import generate_password_hash, check_password_hash
from services.ai_service import AIService
from services.youtube_service import get_lesson_video, get_lesson_videos_list, search_youtube_api

class ComprehensiveIntelliCourseTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['DEMO_MODE'] = True
        self.client = app.test_client()

        with app.app_context():
            db.create_all()

            # Ensure primary test user
            self.user1 = User.query.filter_by(email='user1_comprehensive@intellicourse.ai').first()
            if not self.user1:
                self.user1 = User(
                    name='User One',
                    email='user1_comprehensive@intellicourse.ai',
                    password_hash=generate_password_hash('Password123!')
                )
                db.session.add(self.user1)
                db.session.commit()

            # Ensure secondary test user for access control testing
            self.user2 = User.query.filter_by(email='user2_comprehensive@intellicourse.ai').first()
            if not self.user2:
                self.user2 = User(
                    name='User Two',
                    email='user2_comprehensive@intellicourse.ai',
                    password_hash=generate_password_hash('Password123!')
                )
                db.session.add(self.user2)
                db.session.commit()

            # Ensure User 1 has a course
            self.course1 = Course.query.filter_by(user_id=self.user1.id, title='User1 Python Specialization').first()
            if not self.course1:
                self.course1 = Course(
                    user_id=self.user1.id,
                    title='User1 Python Specialization',
                    topic='Python',
                    skill_level='Beginner'
                )
                db.session.add(self.course1)
                db.session.flush()

                mod = Module(course_id=self.course1.id, position=1, title='Core Basics')
                db.session.add(mod)
                db.session.flush()

                self.lesson1 = Lesson(
                    module_id=mod.id,
                    position=1,
                    title='Introduction to Variables',
                    explanation='Variables store values in computer memory.',
                    concepts_json=json.dumps(['Variables', 'Memory']),
                    coding_exercise_json=json.dumps({'starter_code': 'x = 10'})
                )
                db.session.add(self.lesson1)
                db.session.flush()

                # Assessment
                self.assessment1 = Assessment(module_id=mod.id, title='Basics Checkpoint', passing_score=70)
                db.session.add(self.assessment1)
                db.session.flush()

                q1 = Question(
                    assessment_id=self.assessment1.id,
                    position=1,
                    prompt='What is x in x = 5?',
                    options_json=json.dumps(['Variable', 'Loop', 'Class', 'Function']),
                    correct_index=0,
                    explanation='x is a variable'
                )
                db.session.add(q1)
                db.session.commit()
            else:
                self.lesson1 = self.course1.modules[0].lessons[0]
                self.assessment1 = self.course1.modules[0].assessment

    def test_01_user_registration_new_user(self):
        """Verify new users can register and their passwords are secure."""
        with app.app_context():
            old = User.query.filter_by(email='brand_new_student@intellicourse.ai').first()
            if old:
                db.session.delete(old)
                db.session.commit()

        res = self.client.post('/register', data={
            'name': 'New Student',
            'email': 'brand_new_student@intellicourse.ai',
            'password': 'StrongPassword2026!'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        
        with app.app_context():
            u = User.query.filter_by(email='brand_new_student@intellicourse.ai').first()
            self.assertIsNotNone(u)
            self.assertTrue(check_password_hash(u.password_hash, 'StrongPassword2026!'))

    def test_02_user_registration_duplicate_email(self):
        """Verify duplicate email registration displays exact required error."""
        res = self.client.post('/register', data={
            'name': 'Duplicate Attempt',
            'email': 'user1_comprehensive@intellicourse.ai',
            'password': 'SomePassword123'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'This account already exists. Please log in.', res.data)

    def test_03_login_unregistered_email(self):
        """Verify login with unregistered email displays exact required message."""
        res = self.client.post('/login', data={
            'email': 'nonexistent_account_12345@intellicourse.ai',
            'password': 'AnyPassword123!'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'No account found. Please create an account.', res.data)

    def test_04_login_incorrect_password(self):
        """Verify login with incorrect password displays clear error."""
        res = self.client.post('/login', data={
            'email': 'user1_comprehensive@intellicourse.ai',
            'password': 'TotallyWrongPassword'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Incorrect password. Please try again.', res.data)

    def test_05_login_valid_credentials(self):
        """Verify login with valid credentials succeeds and redirects to dashboard."""
        res = self.client.post('/login', data={
            'email': 'user1_comprehensive@intellicourse.ai',
            'password': 'Password123!'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'User One', res.data)

    def test_06_access_control_isolation(self):
        """Verify User 2 cannot access User 1 private courses or lessons."""
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user2.id)
            sess['_fresh'] = True

        # Attempt to access User 1 course preview
        res = self.client.get(f'/courses/{self.course1.id}')
        self.assertEqual(res.status_code, 404)

        # Attempt to access User 1 lesson
        res_lesson = self.client.get(f'/lessons/{self.lesson1.id}')
        self.assertEqual(res_lesson.status_code, 404)

    def test_07_lesson_completion_and_progress_tracking(self):
        """Verify /api/lessons/<id>/complete marks lesson and recalculates progress."""
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user1.id)
            sess['_fresh'] = True

        res = self.client.post(f'/api/lessons/{self.lesson1.id}/complete')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['ok'])
        self.assertTrue(data['completed'])
        self.assertEqual(data['progress']['percent'], 100)

        # Verify database persistence
        with app.app_context():
            prog = LessonProgress.query.filter_by(user_id=self.user1.id, lesson_id=self.lesson1.id).first()
            self.assertIsNotNone(prog)
            self.assertTrue(prog.completed)

    def test_08_assessment_submission_and_scoring(self):
        """Verify assessment evaluation, score calculation, and attempt record."""
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user1.id)
            sess['_fresh'] = True

        with app.app_context():
            assessment = db.session.get(Assessment, self.assessment1.id)
            q = assessment.questions[0]
            q_id = q.id
            correct_idx = q.correct_index

        res = self.client.post(f'/api/assessments/{self.assessment1.id}/submit', json={
            'answers': {str(q_id): correct_idx}
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['score'], 1)
        self.assertEqual(data['percentage'], 100)
        self.assertTrue(data['passed'])

    def test_09_course_completion_and_certificate_removal(self):
        """Verify certificate feature is removed and course completion functions cleanly."""
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user1.id)
            sess['_fresh'] = True

        # First complete the lesson
        self.client.post(f'/api/lessons/{self.lesson1.id}/complete')

        # And submit the passing assessment
        with app.app_context():
            assessment = db.session.get(Assessment, self.assessment1.id)
            q = assessment.questions[0]
            q_id = q.id
            correct_idx = q.correct_index
        self.client.post(f'/api/assessments/{self.assessment1.id}/submit', json={
            'answers': {str(q_id): correct_idx}
        })

        # Verify certificate endpoint is completely removed (returns 404)
        res = self.client.post(f'/api/courses/{self.course1.id}/certificate')
        self.assertEqual(res.status_code, 404)

        # Verify course completion screen renders cleanly without certificates
        res_comp = self.client.get(f'/courses/{self.course1.id}/completion')
        self.assertEqual(res_comp.status_code, 200)
        content = res_comp.get_data(as_text=True)
        self.assertNotIn('Claim Official Certificate', content)
        self.assertNotIn('certificate_id', content)

    def test_10_youtube_service_handles_quota_and_empty(self):
        """Verify YouTube service returns structured quota_exceeded or fallback data gracefully."""
        # Test with empty API key
        video = get_lesson_video('Advanced Loops', 'Python Programming', api_key='', lesson_index=1)
        self.assertIsNotNone(video)
        self.assertIn(video['source'], ['verified_curated', 'search_fallback'])

        # Test video list API
        vlist = get_lesson_videos_list('Functions', 'Python', api_key='')
        self.assertEqual(vlist['status'], 'no_key')
        self.assertTrue(len(vlist['videos']) > 0)

    def test_11_ai_tutor_answers_with_course_context(self):
        """Verify AI Tutor receives and references course context."""
        ai = AIService(api_key='', demo=True)
        ctx = {
            'course_title': 'Python Programming Masterclass',
            'course_topic': 'Python',
            'title': 'Variables and Expressions',
            'explanation': 'Variables reference stored memory values in Python.'
        }
        answer, source = ai.tutor('What is a variable?', ctx, mode='explain-simply')
        self.assertEqual(source, 'demo')
        self.assertIn('variable', answer.lower())

if __name__ == '__main__':
    unittest.main()
