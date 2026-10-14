import unittest
from services.ai_service import AIService
from services.youtube_service import get_lesson_video, get_lesson_videos_list

class AIServiceTests(unittest.TestCase):
    def setUp(self):
        self.ai = AIService(api_key='', demo=True)

    def test_demo_conversation_collects_preferences(self):
        state = {}
        replies = []
        for message in ['Python for Data Science in 30 days', 'Beginner', '30 days', '1 hour', 'a mixture of both']:
            result = self.ai.reply(state, message, replies)
            state = result['state']
            replies.append({'role': 'user', 'content': message})
        self.assertEqual(state['topic'], 'Python for Data Science in 30 days')
        self.assertEqual(state['skill_level'], 'Beginner')
        self.assertTrue(result['ready'])
        self.assertEqual(state['learning_style'], 'Mixed')

    def test_python_course_validates_and_has_capstone(self):
        payload = self.ai._sample_course({'topic': 'Python Basics', 'duration_days': 30, 'daily_minutes': 60, 'skill_level': 'Beginner'})
        self.assertTrue(self.ai.validate_course(payload))
        self.assertEqual(len(payload['modules']), 6)
        self.assertGreaterEqual(len(payload['modules'][0]['lessons']), 3)
        self.assertIn('capstone_project', payload)
        self.assertTrue(len(payload['capstone_project']['title']) > 0)
        self.assertTrue(len(payload['capstone_project']['tasks']) >= 3)

    def test_machine_learning_course_is_distinct_from_python(self):
        py_payload = self.ai._sample_course({'topic': 'Python Programming', 'duration_days': 30, 'daily_minutes': 60, 'skill_level': 'Beginner'})
        ml_payload = self.ai._sample_course({'topic': 'Machine Learning', 'duration_days': 45, 'daily_minutes': 60, 'skill_level': 'Intermediate'})
        self.assertTrue(self.ai.validate_course(ml_payload))
        # Ensure curricula are substantially different
        self.assertNotEqual(py_payload['title'], ml_payload['title'])
        self.assertNotEqual(py_payload['modules'][0]['title'], ml_payload['modules'][0]['title'])
        self.assertIn('Preprocessing', ml_payload['modules'][0]['title'])

    def test_generic_custom_topic_generates_valid_course(self):
        custom = self.ai._sample_course({'topic': 'Quantum Computing Foundations', 'duration_days': 21, 'daily_minutes': 45, 'skill_level': 'Beginner'})
        self.assertTrue(self.ai.validate_course(custom))
        self.assertEqual(len(custom['modules']), 6)

    def test_rejects_invalid_course(self):
        self.assertFalse(self.ai.validate_course({'title': 'Bad', 'modules': []}))
        self.assertFalse(self.ai.validate_course(None))
        self.assertFalse(self.ai.validate_course({'title': 'No modules'}))

    def test_youtube_service_retrieves_curated_video(self):
        video = get_lesson_video('Variables and Expressions', 'Python Basics', api_key='', lesson_index=0)
        self.assertIsNotNone(video)
        self.assertEqual(video['source'], 'verified_curated')
        self.assertTrue(video['youtube_url'].startswith('https://www.youtube.com/watch?v='))
        self.assertTrue(video['embed_url'].startswith('https://www.youtube.com/embed/'))

    def test_youtube_service_fallback_on_custom_topic(self):
        video = get_lesson_video('Introduction to String Theory', 'String Theory', api_key='', lesson_index=0)
        self.assertIsNotNone(video)
        self.assertEqual(video['source'], 'search_fallback')
        self.assertIn('youtube.com/results?search_query=', video['youtube_url'])

    def test_tutor_answers_with_modes(self):
        ans1, src1 = self.ai.tutor('How do functions work?', {'title': 'Python Functions', 'course_title': 'Python'}, mode='explain-simply')
        self.assertEqual(src1, 'demo')
        self.assertIn('Explanation', ans1)

        ans2, src2 = self.ai.tutor('Why do I get a TypeError?', {'title': 'Python Data Types', 'course_title': 'Python'}, mode='debug-code', code_snippet='x = "5" + 2')
        self.assertEqual(src2, 'demo')
        self.assertIn('Debugging', ans2)

if __name__ == '__main__':
    unittest.main()
