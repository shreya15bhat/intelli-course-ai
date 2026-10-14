import json
from models import db, Course, Module, Lesson, Assessment, Question
from services.youtube_service import get_lesson_video

def save_course(user, prefs, payload, youtube_api_key=""):
    """Persist generated course and its curriculum into relational database."""
    description = payload.get('description', '')
    sources = payload.get('sources', [])
    if sources:
        links = '\n\nOfficial Documentation & References:\n' + '\n'.join(
            f"- {str(item.get('title','Reference'))[:200]}: {str(item.get('url',''))[:500]}"
            for item in sources if isinstance(item, dict) and str(item.get('url','')).startswith('https://')
        )
        description = (description + links)[:5000]

    course = Course(
        user_id=user.id,
        title=str(payload.get('title', 'Comprehensive Course'))[:180],
        description=description,
        topic=str(prefs.get('topic', ''))[:180],
        skill_level=str(prefs.get('skill_level', 'Beginner'))[:40],
        duration_days=int(prefs.get('duration_days', 30)),
        daily_minutes=int(prefs.get('daily_minutes', 60)),
        learning_style=str(prefs.get('learning_style', 'Mixed'))[:80],
        learning_goal=str(prefs.get('learning_goal', ''))[:500],
        desired_outcome=str(prefs.get('desired_outcome', ''))[:500],
        language=str(prefs.get('language', 'English'))[:60],
        objectives_json=json.dumps(payload.get('objectives', [])),
        prerequisites_json=json.dumps(payload.get('prerequisites', [])),
        capstone_project_json=json.dumps(payload.get('capstone_project', {}))
    )
    db.session.add(course)
    db.session.flush()

    lesson_counter = 0
    seen_video_ids = set()
    for mi, md in enumerate(payload.get('modules', []), 1):
        module = Module(
            course_id=course.id,
            position=mi,
            title=str(md.get('title', f'Module {mi}'))[:180],
            description=str(md.get('description', ''))[:3000],
            duration_minutes=max(15, int(md.get('duration_minutes', 60))),
            objectives_json=json.dumps(md.get('objectives', []))
        )
        db.session.add(module)
        db.session.flush()

        for li, ld in enumerate(md.get('lessons', []), 1):
            lesson_counter += 1
            ltitle = str(ld.get('title', f'Lesson {li}'))[:180]
            # Retrieve or attach verified educational video
            video_data = ld.get('video')
            if not video_data or not isinstance(video_data, dict) or not video_data.get('video_id'):
                video_data = get_lesson_video(ltitle, course.topic, youtube_api_key, lesson_index=lesson_counter, seen_ids=seen_video_ids)
            if video_data and video_data.get('video_id'):
                seen_video_ids.add(video_data['video_id'])

            lesson = Lesson(
                module_id=module.id,
                position=li,
                title=ltitle,
                explanation=str(ld.get('explanation', '')),
                concepts_json=json.dumps(ld.get('concepts', [])),
                examples_json=json.dumps(ld.get('examples', [])),
                exercises_json=json.dumps(ld.get('exercises', [])),
                takeaways_json=json.dumps(ld.get('takeaways', [])),
                coding_exercise_json=json.dumps(ld.get('coding_exercise', {})),
                video_json=json.dumps(video_data or {})
            )
            db.session.add(lesson)

        # Assessment / Checkpoint for the module
        questions = md.get('questions', [])
        if questions:
            is_final_module = (mi == len(payload.get('modules', [])))
            assessment_title = f"{md['title']} Checkpoint" if not is_final_module else f"Final Comprehensive Assessment"
            assessment = Assessment(
                module_id=module.id,
                title=assessment_title[:180],
                is_final=is_final_module,
                passing_score=70
            )
            db.session.add(assessment)
            db.session.flush()

            for qi, q in enumerate(questions, 1):
                db.session.add(Question(
                    assessment_id=assessment.id,
                    position=qi,
                    prompt=str(q.get('prompt', 'Question')),
                    options_json=json.dumps(q.get('options', [])),
                    correct_index=int(q.get('correct_index', 0)),
                    explanation=str(q.get('explanation', ''))
                ))

    db.session.commit()
    return course
