import json
from models import db, Course, Module, Lesson, Assessment, Question

def save_course(user, prefs, payload):
    description = payload.get('description','')
    sources = payload.get('sources', [])
    if sources:
        links = '\n\nLearning references (verify the current documentation):\n' + '\n'.join(
            f"- {str(item.get('title','Reference'))[:200]}: {str(item.get('url',''))[:500]}"
            for item in sources if isinstance(item, dict) and str(item.get('url','')).startswith('https://')
        )
        description = (description + links)[:5000]
    course = Course(user_id=user.id, title=payload['title'][:180], description=description, topic=prefs.get('topic',''), skill_level=str(prefs.get('skill_level','Beginner'))[:40], duration_days=int(prefs.get('duration_days',30)), daily_minutes=int(prefs.get('daily_minutes',60)), learning_style=str(prefs.get('learning_style','Mixed'))[:80], learning_goal=prefs.get('learning_goal',''), desired_outcome=prefs.get('desired_outcome',''), language=prefs.get('language','English'), objectives_json=json.dumps(payload.get('objectives',[])))
    db.session.add(course); db.session.flush()
    for mi, md in enumerate(payload['modules'],1):
        module = Module(course_id=course.id, position=mi, title=md['title'][:180], description=md.get('description',''), duration_minutes=max(1,int(md.get('duration_minutes',60))), objectives_json=json.dumps(md.get('objectives',[])))
        db.session.add(module); db.session.flush()
        for li, ld in enumerate(md.get('lessons',[]),1):
            db.session.add(Lesson(module_id=module.id, position=li, title=ld['title'][:180], explanation=ld.get('explanation',''), concepts_json=json.dumps(ld.get('concepts',[])), examples_json=json.dumps(ld.get('examples',[])), exercises_json=json.dumps(ld.get('exercises',[])), takeaways_json=json.dumps(ld.get('takeaways',[]))))
        if md.get('questions'):
            assessment=Assessment(module_id=module.id,title=f'{md["title"]} Checkpoint'); db.session.add(assessment); db.session.flush()
            for qi,q in enumerate(md['questions'],1):
                db.session.add(Question(assessment_id=assessment.id,position=qi,prompt=q['prompt'],options_json=json.dumps(q['options']),correct_index=int(q['correct_index']),explanation=q.get('explanation','')))
    db.session.commit()
    return course
