import unittest
from services.ai_service import AIService

class AIServiceTests(unittest.TestCase):
    def setUp(self):
        self.ai = AIService(api_key='', demo=True)

    def test_demo_conversation_collects_preferences(self):
        state={}
        replies=[]
        for message in ['Python for Data Science in 30 days','Beginner','30 days','1 hour','a mixture of both']:
            result=self.ai.reply(state,message,replies)
            state=result['state']; replies.append({'role':'user','content':message})
        self.assertEqual(state['topic'],'Python for Data Science in 30 days')
        self.assertEqual(state['skill_level'],'Beginner')
        self.assertTrue(result['ready'])
        self.assertEqual(state['learning_style'],'Mixed')

    def test_sample_course_validates(self):
        payload=self.ai._sample_course({'topic':'Python for Data Science','duration_days':30,'daily_minutes':60,'skill_level':'Beginner'})
        self.assertTrue(self.ai.validate_course(payload))
        self.assertEqual(len(payload['modules']),6)
        self.assertGreaterEqual(len(payload['modules'][0]['lessons']),3)

    def test_rejects_invalid_course(self):
        self.assertFalse(self.ai.validate_course({'title':'Bad','modules':[]}))

if __name__=='__main__': unittest.main()
