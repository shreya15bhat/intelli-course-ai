import unittest
import json
from app import app, db
from models import User, Course, Module, Lesson, CourseTutorMessage

class APIEndpointsTest(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['DEMO_MODE'] = True
        self.client = app.test_client()

        with app.app_context():
            db.create_all()
            self.user = User.query.filter_by(email='tester@coursemind.ai').first()
            if not self.user:
                self.user = User(
                    name='Test Scholar',
                    email='tester@coursemind.ai',
                    password_hash='pbkdf2:sha256:test'
                )
                db.session.add(self.user)
                db.session.commit()

            # Ensure test course 1 (Python Basics)
            self.course1 = Course.query.filter_by(user_id=self.user.id, title='Python Basics').first()
            if not self.course1:
                self.course1 = Course(
                    user_id=self.user.id,
                    title='Python Basics',
                    topic='Python Basics',
                    skill_level='Beginner'
                )
                db.session.add(self.course1)
                db.session.flush()

                mod1 = Module(course_id=self.course1.id, position=1, title='Variables and Basics')
                db.session.add(mod1)
                db.session.flush()

                self.lesson1 = Lesson(
                    module_id=mod1.id,
                    position=1,
                    title='Variables and Data Types',
                    explanation='A variable in Python is a named reference that points to a value stored in memory.',
                    concepts_json=json.dumps(['Variables', 'Assignment', 'Data Types']),
                    examples_json=json.dumps(['age = 20\nprint(age)'])
                )
                db.session.add(self.lesson1)
                db.session.commit()
            else:
                self.lesson1 = self.course1.modules[0].lessons[0]

            # Ensure test course 2 (SQL Mastery) for isolation test
            self.course2 = Course.query.filter_by(user_id=self.user.id, title='SQL Mastery').first()
            if not self.course2:
                self.course2 = Course(
                    user_id=self.user.id,
                    title='SQL Mastery',
                    topic='SQL',
                    skill_level='Beginner'
                )
                db.session.add(self.course2)
                db.session.flush()

                mod2 = Module(course_id=self.course2.id, position=1, title='Query Fundamentals')
                db.session.add(mod2)
                db.session.flush()

                self.lesson2 = Lesson(
                    module_id=mod2.id,
                    position=1,
                    title='Introduction to SELECT',
                    explanation='SELECT statements retrieve data from relational database tables.'
                )
                db.session.add(self.lesson2)
                db.session.commit()
            else:
                self.lesson2 = self.course2.modules[0].lessons[0]

    def test_health_endpoint(self):
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('demo_mode', data)

    def test_videos_endpoint(self):
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user.id)
            sess['_fresh'] = True

        res = self.client.get(f'/api/lessons/{self.lesson1.id}/videos')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('status', data)
        self.assertIn('videos', data)
        self.assertTrue(len(data['videos']) > 0)
        first_video = data['videos'][0]
        self.assertIn('title', first_video)
        self.assertIn('channel', first_video)
        self.assertIn('youtube_url', first_video)
        self.assertIn('embed_url', first_video)

    def test_tutor_answers_define_variables(self):
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user.id)
            sess['_fresh'] = True

        res = self.client.post('/api/ai/tutor', json={
            'question': 'Define variables.',
            'lesson_id': self.lesson1.id,
            'mode': 'answer'
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['ok'])
        self.assertIn('answer', data)
        self.assertTrue(len(data['answer']) > 0)
        # Should address variables directly
        self.assertIn('variable', data['answer'].lower())

    def test_tutor_rejects_empty_question(self):
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user.id)
            sess['_fresh'] = True

        res = self.client.post('/api/ai/tutor', json={
            'question': '   ',
            'lesson_id': self.lesson1.id
        })
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn('error', data)

    def test_tutor_isolation_between_courses(self):
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user.id)
            sess['_fresh'] = True

        # Ask in Course 1
        res1 = self.client.post('/api/ai/tutor', json={
            'question': 'How do Python variables work?',
            'lesson_id': self.lesson1.id
        })
        self.assertEqual(res1.status_code, 200)

        # Ask in Course 2
        res2 = self.client.post('/api/ai/tutor', json={
            'question': 'What does SELECT do?',
            'lesson_id': self.lesson2.id
        })
        self.assertEqual(res2.status_code, 200)

        # Check history for Course 1
        hist1 = self.client.get(f'/api/lessons/{self.lesson1.id}/tutor/history').get_json()
        # Check history for Course 2
        hist2 = self.client.get(f'/api/lessons/{self.lesson2.id}/tutor/history').get_json()

        # Messages must not be mixed
        c1_texts = [m['content'] for m in hist1['messages']]
        c2_texts = [m['content'] for m in hist2['messages']]
        self.assertTrue(any('Python variables' in t for t in c1_texts))
        self.assertFalse(any('Python variables' in t for t in c2_texts))
        self.assertTrue(any('SELECT' in t for t in c2_texts))

    def test_tutor_clear_history(self):
        with self.client.session_transaction() as sess:
            sess['_user_id'] = str(self.user.id)
            sess['_fresh'] = True

        # Send a message
        self.client.post('/api/ai/tutor', json={
            'question': 'Temporary test question',
            'lesson_id': self.lesson1.id
        })

        # Clear history
        clear_res = self.client.post(f'/api/courses/{self.course1.id}/tutor/clear')
        self.assertEqual(clear_res.status_code, 200)

        # Check history is now empty
        hist = self.client.get(f'/api/lessons/{self.lesson1.id}/tutor/history').get_json()
        self.assertEqual(len(hist['messages']), 0)

if __name__ == '__main__':
    unittest.main()
