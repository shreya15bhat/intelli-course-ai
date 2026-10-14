"""Google Gemini API service for course generation and intelligent course tutoring.

Uses the official current Google Gen AI SDK (google-genai) with Google Search
grounding where supported. When GEMINI_API_KEY is not configured or in DEMO_MODE,
provides transparent, topic-specific high quality courses with code examples,
exercises, test cases, quizzes, and capstone projects.
"""
import json
import re
import time
from typing import Dict, Any, Tuple, Optional, List

TUTOR_SYSTEM_INSTRUCTIONS = """You are CourseMind AI Tutor, an accurate, patient, course-aware teaching assistant.

Your primary task is to answer the learner's exact question directly before adding additional explanation.

Understand the question carefully. Identify the concept, code, error, or doubt being asked about. Use the selected course and lesson content as your primary context.

Give the correct answer in simple, clear language appropriate to the learner's level. Explain the reasoning and provide a relevant example when useful.

If the learner asks for a definition, provide the definition first. If they ask for a difference, compare the concepts directly. If they ask for code, provide complete, syntactically correct code and explain the expected output. If they ask why something happens, explain the underlying reason. If they report an error, diagnose the actual error rather than providing an unrelated tutorial.

Answer questions about any module in the selected course, not just the current lesson. Use the saved course curriculum to identify relevant lessons.

Do not repeat generic advice such as 'review the lesson', 'click Run Code', or 'check the key concepts' instead of answering the question.

Do not invent facts, nonexistent functions, code outputs, or course content. If the saved lesson is incomplete, use your relevant programming or subject knowledge to provide a correct answer and clearly distinguish additional knowledge from the material explicitly present in the course.

If the question is ambiguous, ask one brief clarification. If you do not know the answer confidently, explain the uncertainty honestly.

Stay focused on the learner's question. Do not overwhelm the learner with unrelated information."""

class AIService:
    def __init__(self, api_key: str = '', model: str = 'gemini-3.1-flash-lite', demo: bool = False):
        self.api_key = (api_key or '').strip()
        self.model = model or 'gemini-3.1-flash-lite'
        # Run demo mode if explicitly requested (DEMO_MODE=true) or if no API key is provided
        self.demo = bool(demo) or not bool(self.api_key)
        self.client = None
        self.types = None
        self.init_error = None

        if self.api_key and not self.demo:
            try:
                from google import genai
                from google.genai import types
                self.types = types
                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                self.client = None
                self.init_error = str(exc)

    def verify_model_access(self) -> Dict[str, Any]:
        """Verify whether configured model is supported and accessible."""
        if not self.api_key:
            return {"accessible": False, "reason": "No GEMINI_API_KEY configured."}
        if not self.client:
            return {"accessible": False, "reason": f"Client initialization failed: {self.init_error}"}
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents="ping"
            )
            return {"accessible": True, "model": self.model}
        except Exception as exc:
            return {"accessible": False, "model": self.model, "error": str(exc)}

    def _generate_text(
        self,
        prompt: str,
        use_google_search: bool = False,
        as_json: bool = False
    ) -> str:
        """Call Gemini model with automatic retry, fast model fallback, and explicit error propagation."""
        if not self.client:
            err_details = f": {self.init_error}" if self.init_error else ""
            raise RuntimeError(f"Gemini API key is missing or client failed to initialize{err_details}. Please check GEMINI_API_KEY in the backend .env file.")

        # Prioritize supported fast models only
        supported_candidates = [self.model, 'gemini-3.1-flash-lite', 'gemini-3.1-flash-lite-preview', 'gemini-3-flash-preview']
        models_to_try = []
        for m in supported_candidates:
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_error = None
        for m in models_to_try:
            # Limited retries (max 2 attempts) for transient failures only
            for attempt in range(2):
                try:
                    if self.types:
                        http_opts = self.types.HttpOptions(timeout=30000)
                        config_kwargs = {
                            'temperature': 0.2,
                            'http_options': http_opts
                        }
                        if as_json:
                            config_kwargs['response_mime_type'] = 'application/json'
                        if use_google_search:
                            config_kwargs['tools'] = [self.types.Tool(google_search=self.types.GoogleSearch())]
                        config = self.types.GenerateContentConfig(**config_kwargs)
                    else:
                        config = {'temperature': 0.2}

                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=config
                    )
                    text = (response.text or '').strip()
                    if text:
                        return text
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc).lower()

                    # Permanent errors: do not retry
                    if any(term in err_str for term in ['api_key_invalid', 'permission_denied', 'unauthenticated', '401', '403']):
                        raise RuntimeError(f"Gemini API authentication failed: {exc}")
                    if '404' in err_str or 'not found' in err_str or 'no longer available' in err_str:
                        # Model not available, immediately switch to next candidate model
                        break
                    if '400' in err_str or 'invalid_argument' in err_str:
                        # Malformed request, retrying the exact same request won't succeed
                        break
                    if '429' in err_str or 'resource_exhausted' in err_str or 'quota' in err_str:
                        # Quota exceeded: do not spam repeated retries
                        raise RuntimeError("Gemini API rate limit or quota exceeded. Please wait a moment before trying again.")

                    # Transient server / network error (500, 503, timeout): retry with backoff
                    if attempt == 0:
                        time.sleep(1.5)
                        continue

        raise RuntimeError(f"Gemini API request failed: {last_error}")

    @staticmethod
    def _parse_json(raw: Optional[str]) -> Optional[Dict]:
        """Extract and parse JSON from model output, handling Markdown fences and surrounding text."""
        if not raw:
            return None
        cleaned = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw.strip(), flags=re.IGNORECASE).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except Exception:
                    pass
        return None

    @staticmethod
    def _clean_state(state: Dict) -> Dict:
        s = dict(state or {})
        for key in ('topic', 'learning_goal', 'desired_outcome', 'language', 'learning_style', 'skill_level'):
            if isinstance(s.get(key), str):
                s[key] = s[key].strip()[:500]
        for key, default, maximum in [('duration_days', 30, 365), ('daily_minutes', 60, 1440)]:
            try:
                val = int(s.get(key, default))
                s[key] = max(1, min(maximum, val))
            except (ValueError, TypeError):
                s[key] = default
        if s.get('skill_level'):
            lvl = str(s['skill_level']).lower()
            if 'adv' in lvl:
                s['skill_level'] = 'Advanced'
            elif 'inter' in lvl:
                s['skill_level'] = 'Intermediate'
            else:
                s['skill_level'] = 'Beginner'
        else:
            s['skill_level'] = 'Beginner'

        if s.get('learning_style'):
            sty = str(s['learning_style']).lower()
            if 'pract' in sty:
                s['learning_style'] = 'Practical'
            elif 'theory' in sty:
                s['learning_style'] = 'Theory'
            else:
                s['learning_style'] = 'Mixed'
        else:
            s['learning_style'] = 'Mixed'
        return s

    def reply(self, state: Dict, user_message: str, history: List[Dict]) -> Dict:
        """Conversational course profile planning."""
        cleaned_state = self._clean_state(state)
        if self.client:
            prompt = f'''You are CourseMind AI's course-planning tutor. Help the learner plan a comprehensive course.
Extract preferences strictly when the user provides them:
Return ONLY valid JSON matching this exact structure:
{{"state":{{"topic":"","learning_goal":"","skill_level":"Beginner|Intermediate|Advanced","duration_days":30,"daily_minutes":60,"learning_style":"Theory|Practical|Mixed","desired_outcome":"","language":"English"}},"reply":"Helpful conversational reply","ready":false}}

Required fields for ready=true: topic, skill_level, duration_days, daily_minutes, learning_style.
Current profile: {json.dumps(cleaned_state, ensure_ascii=False)}
Recent chat: {json.dumps(history[-8:], ensure_ascii=False)}
New message from user: {user_message}'''
            data = self._parse_json(self._generate_text(prompt, use_google_search=False))
            if isinstance(data, dict) and isinstance(data.get('state'), dict) and isinstance(data.get('reply'), str):
                data['state'] = self._clean_state({**cleaned_state, **data['state']})
                data['ready'] = self._is_ready(data['state'])
                return data
        return self._demo_reply(cleaned_state, user_message)

    @staticmethod
    def _is_ready(s: Dict) -> bool:
        return bool(s.get('topic') and s.get('skill_level') and s.get('duration_days') and s.get('daily_minutes') and s.get('learning_style'))

    def generate_course(self, prefs: Dict) -> Tuple[Dict, str]:
        """Generate a complete structured topic-specific course.
        
        Returns (course_payload, source) where source is 'gemini_google_search', 'gemini', or 'demo'.
        """
        p = self._clean_state(prefs)
        topic = p.get('topic', 'General Learning').strip()
        level = p.get('skill_level', 'Beginner')
        duration = p.get('duration_days', 30)
        daily_min = p.get('daily_minutes', 60)
        style = p.get('learning_style', 'Mixed')

        if self.client:
            prompt = f'''You are a senior curriculum architect designing an online masterclass on "{topic}" ({level} level, {duration} days, {daily_min} min/day, style: {style}).
Build a complete, thorough, pedagogically sound course curriculum inspired by Coursera.
Do NOT use generic filler text. Cover actual concepts, definitions, syntax, worked code examples, exercises, module quizzes, and a realistic capstone project.

Return ONLY one valid JSON object with this exact structure:
{{
  "title": "{topic}: Comprehensive {level} Specialization",
  "description": "In-depth course overview explaining target outcomes, core methodology, and real-world applications.",
  "objectives": ["Objective 1", "Objective 2", "Objective 3"],
  "prerequisites": ["Basic computer literacy"],
  "capstone_project": {{
    "title": "Topic-Specific Capstone Project Title",
    "description": "Real-world project scenario and problem statement.",
    "tasks": ["Task 1: Setup and schema", "Task 2: Core implementation", "Task 3: Testing and validation"],
    "starter_code": "# Starter template\\n",
    "deliverables": "Working application, tests, and documentation."
  }},
  "sources": [{{"title": "Official Documentation", "url": "https://docs.python.org/3/"}}],
  "modules": [
    {{
      "title": "Module 1 Title",
      "description": "Detailed explanation of what this module covers and why it matters.",
      "duration_minutes": 90,
      "objectives": ["Module Objective 1", "Module Objective 2"],
      "lessons": [
        {{
          "title": "Lesson 1.1 Title",
          "explanation": "Clear, thorough pedagogical explanation teaching the concept. Defines terms, mechanics, syntax, and common pitfalls.",
          "concepts": ["Concept 1 with description", "Concept 2 with description"],
          "examples": ["Code or worked example with explanation:\\n```python\\n# code here\\n```\\nExpected Output:\\n```\\noutput here\\n```"],
          "exercises": ["Step-by-step guided practice activity"],
          "takeaways": ["Core takeaway 1", "Core takeaway 2"],
          "coding_exercise": {{
            "language": "python",
            "instructions": "Write code to accomplish this specific task.",
            "starter_code": "def solution():\\n    # Your code here\\n    pass\\n",
            "solution": "def solution():\\n    return True\\n",
            "test_cases": [
              {{"input": "test_input_1", "expected": "expected_output_1"}}
            ]
          }}
        }}
      ],
      "questions": [
        {{
          "prompt": "Clear conceptual question testing module understanding",
          "options": ["Correct option", "Distractor A", "Distractor B", "Distractor C"],
          "correct_index": 0,
          "explanation": "Detailed explanation of why the correct answer is right."
        }},
        {{
          "prompt": "Second question testing practical application",
          "options": ["Distractor A", "Correct option", "Distractor B", "Distractor C"],
          "correct_index": 1,
          "explanation": "Detailed explanation."
        }},
        {{
          "prompt": "Third question testing debugging or syntax",
          "options": ["Distractor A", "Distractor B", "Correct option", "Distractor C"],
          "correct_index": 2,
          "explanation": "Detailed explanation."
        }}
      ]
    }}
  ]
}}

Requirements:
- Exactly 4 modules logically sequenced from foundations to practical application.
- Exactly 2 high-quality lessons per module (8 lessons total).
- Each lesson explanation must be a clear 1-2 paragraph guide explaining concepts, syntax, and common pitfalls.
- Exactly 3 questions per module with 4 distinct options and valid correct_index (0-3).
- For programming topics ({topic}), provide valid starter code, instructions, and test cases in coding_exercise.
- Return raw valid JSON only matching the schema.'''
            
        if self.client and not self.demo:
            try:
                raw_response = self._generate_text(prompt, use_google_search=False, as_json=True)
                data = self._parse_json(raw_response)
                if data and self.validate_course(data):
                    data['sources'] = self._valid_sources(data.get('sources', []))
                    source_label = 'gemini_google_search' if data.get('sources') else 'gemini'
                    return data, source_label
                elif data:
                    raise ValueError("Generated course content did not meet required curriculum structure. Please retry.")
                else:
                    raise ValueError("AI model returned an unparseable response. Please retry.")
            except Exception as exc:
                err_msg = str(exc)
                if "Live AI course generation failed" in err_msg:
                    raise
                raise RuntimeError(f"Live AI course generation failed: {err_msg}")

        # Deterministic rich fallback strictly for offline/demo mode (when DEMO_MODE=true or no API key)
        sample = self._sample_course(p)
        return sample, 'demo'

    @staticmethod
    def _valid_sources(sources: Any) -> List[Dict]:
        result = []
        if isinstance(sources, list):
            for item in sources:
                if isinstance(item, dict) and isinstance(item.get('title'), str) and isinstance(item.get('url'), str):
                    if item['url'].startswith('https://'):
                        result.append({'title': item['title'][:180], 'url': item['url'][:500]})
        return result[:6]

    @staticmethod
    def validate_course(data: Any) -> bool:
        """Thoroughly validate course JSON structure and completeness."""
        if not isinstance(data, dict):
            return False
        title = data.get('title')
        if not isinstance(title, str) or len(title.strip()) < 3:
            return False
        modules = data.get('modules')
        if not isinstance(modules, list) or len(modules) < 4:
            return False
        
        lesson_count = 0
        for m in modules:
            if not isinstance(m, dict) or not isinstance(m.get('title'), str) or not m['title'].strip():
                return False
            lessons = m.get('lessons')
            if not isinstance(lessons, list) or len(lessons) < 2:
                return False
            lesson_count += len(lessons)
            for l in lessons:
                if not isinstance(l, dict) or not isinstance(l.get('title'), str) or not l['title'].strip():
                    return False
                explanation = l.get('explanation')
                if not isinstance(explanation, str) or len(explanation.strip()) < 40:
                    return False
            questions = m.get('questions')
            if not isinstance(questions, list) or len(questions) < 3:
                return False
            for q in questions:
                if not isinstance(q, dict) or not isinstance(q.get('prompt'), str) or not q['prompt'].strip():
                    return False
                options = q.get('options')
                if not isinstance(options, list) or len(options) != 4 or len({str(o).strip() for o in options}) != 4:
                    return False
                correct = q.get('correct_index')
                if not isinstance(correct, int) or correct not in range(4):
                    return False
        return lesson_count >= 8

    def tutor(
        self,
        question: str,
        lesson_context: Optional[Dict] = None,
        mode: str = 'answer',
        code_snippet: str = '',
        conversation_history: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Course-aware, accurate tutor for the active course and lesson."""
        if self.client:
            ctx = lesson_context or {}
            course_title = ctx.get('course_title', 'Course')
            course_topic = ctx.get('course_topic', '')
            skill_level = ctx.get('skill_level', 'Beginner')
            objectives = ctx.get('course_objectives', [])
            curriculum = ctx.get('curriculum_outline', [])

            module_title = ctx.get('module_title', '')
            lesson_title = ctx.get('title', '')
            explanation = ctx.get('explanation', '')
            concepts = ctx.get('concepts', [])
            examples = ctx.get('examples', [])
            exercises = ctx.get('exercises', [])
            takeaways = ctx.get('takeaways', [])
            starter_code = ctx.get('starter_code', '')

            prev_lesson = ctx.get('previous_lesson')
            next_lesson = ctx.get('next_lesson')
            retrieved = ctx.get('retrieved_curriculum_matches', [])

            # Format curriculum outline
            curriculum_lines = []
            for mod in curriculum:
                mod_t = mod.get('title', 'Module')
                les_titles = ", ".join(mod.get('lessons', []))
                curriculum_lines.append(f"- Module: {mod_t} (Lessons: {les_titles})")
            curriculum_str = "\n".join(curriculum_lines) if curriculum_lines else "Curriculum details in lesson context."

            # Format surrounding lessons
            surrounding_parts = []
            if prev_lesson:
                surrounding_parts.append(f"- Previous Lesson: {prev_lesson.get('title', '')}")
            if next_lesson:
                surrounding_parts.append(f"- Next Lesson: {next_lesson.get('title', '')}")
            surrounding_str = "\n".join(surrounding_parts) if surrounding_parts else "None."

            # Format retrieved lessons (cross-module search)
            retrieved_parts = []
            for r in retrieved:
                r_title = r.get('title', '')
                r_mod = r.get('module_title', '')
                r_expl = r.get('explanation', '')[:400]
                r_conc = ", ".join(r.get('concepts', []))
                retrieved_parts.append(f"- Module: {r_mod} | Lesson: {r_title}\n  Concepts: {r_conc}\n  Summary: {r_expl}")
            retrieved_str = "\n".join(retrieved_parts) if retrieved_parts else "No other module retrieved."

            # Format conversation history
            history_lines = []
            if conversation_history:
                for msg in conversation_history[-8:]:
                    role_tag = "Learner" if msg.get('role') == 'user' else "CourseMind Tutor"
                    content = str(msg.get('content', '')).strip()
                    history_lines.append(f"{role_tag}: {content}")
            history_str = "\n".join(history_lines) if history_lines else "None (New conversation turn)."

            # Mode specific directives
            mode_directives = {
                'explain-simply': 'DIRECTIVE: Explain the requested concept in simple, accessible language with an intuitive everyday analogy. Avoid unnecessary jargon while maintaining technical precision.',
                'step-by-step': 'DIRECTIVE: Provide a logically ordered, step-by-step breakdown or walkthrough.',
                'debug-code': 'DIRECTIVE: Inspect the learner code carefully. Diagnose the exact bug or error, explain WHY it occurred, and provide the complete, corrected code with comments.',
                'give-example': 'DIRECTIVE: Provide a clean, practical, and fully runnable code example with clear annotations and the exact expected output.',
                'answer': 'DIRECTIVE: Answer the learner\'s exact question directly before adding additional explanation.'
            }
            mode_text = mode_directives.get(mode, mode_directives['answer'])

            code_part = f"\n### LEARNER'S CURRENT EDITOR CODE\n```\n{code_snippet}\n```" if code_snippet else ""
            if starter_code and not code_snippet:
                code_part = f"\n### LESSON STARTER CODE\n```\n{starter_code}\n```"

            prompt = f"""{TUTOR_SYSTEM_INSTRUCTIONS}

### COURSE METADATA
- Course Title: {course_title}
- Subject / Topic: {course_topic}
- Target Level: {skill_level}
- Learning Objectives: {json.dumps(objectives, ensure_ascii=False)}

### FULL COURSE CURRICULUM
{curriculum_str}

### CURRENT MODULE & LESSON
- Current Module: {module_title}
- Current Lesson: {lesson_title}
- Lesson Explanation:
{explanation}
- Key Concepts:
{json.dumps(concepts, ensure_ascii=False)}
- Worked Examples:
{json.dumps(examples, ensure_ascii=False)}
- Practice Exercises:
{json.dumps(exercises, ensure_ascii=False)}
- Key Takeaways:
{json.dumps(takeaways, ensure_ascii=False)}

### SURROUNDING LESSON CONTEXT
{surrounding_str}

### RELEVANT LESSONS RETRIEVED FROM CURRICULUM
{retrieved_str}
{code_part}

### RELEVANT CONVERSATION HISTORY (THIS COURSE)
{history_str}

### ACTIVE TUTOR MODE: {mode}
{mode_text}

### LEARNER'S EXACT QUESTION
"{question}"
"""
            answer = self._generate_text(prompt, use_google_search=False)
            return answer, 'gemini'

        if self.api_key and not self.client and not self.demo:
            err_msg = f": {self.init_error}" if self.init_error else ""
            raise RuntimeError(f"Gemini API client failed to initialize{err_msg}. Please check GEMINI_API_KEY in backend .env.")

        # Offline demo mode fallback (when DEMO_MODE=true or no API key configured)
        return self._demo_tutor(question, lesson_context, mode, code_snippet), 'demo'

    @staticmethod
    def _demo_tutor(question: str, lesson: Optional[Dict], mode: str, code: str) -> str:
        title = (lesson or {}).get('title', 'this topic')
        course = (lesson or {}).get('course_title', 'your course')
        
        if "debug" in mode or "error" in question.lower() or "bug" in question.lower():
            return f"""### Code Debugging Guidance ({title})

When debugging code in **{title}**, check these common failure points:
1. **Variable scoping & initialization**: Ensure all variables are declared and initialized before use.
2. **Indentation & Syntax**: Verify brackets, colons, and indentation levels are consistent.
3. **Type mismatches**: Verify whether functions expect integers, strings, or collections.
4. **Boundary conditions**: Test what happens with empty inputs, `0`, `None`, or index bounds.

> **Tip**: Add a `GEMINI_API_KEY` in your `.env` file to enable live AI debugging with automated syntax and semantic analysis!"""

        if mode == 'explain-simply':
            return f"""### Simple Explanation of {title}

Think of **{title}** like building blocks in **{course}**:
- You take an input, apply a clear set of rules or transformations, and produce an expected output.
- Breaking complex logic into single-purpose components keeps your program maintainable and predictable.

> Configure your `GEMINI_API_KEY` in `.env` to unlock live conversational explanations grounded in official documentation!"""

        return f"""### Tutor Note on {title}

You asked: *"{question}"*

In the context of **{title}**:
- Review the key concepts and worked examples in the main lesson panel.
- Test your understanding by modifying the starter code in the interactive code editor and clicking **Run Code**.
- Check the module quiz when you complete all lessons in this module.

*Note: The tutor is currently in Demo Mode. Add a `GEMINI_API_KEY` to `.env` to unlock live AI answers with Google Search grounding.*"""

    def _demo_reply(self, state: Dict, message: str) -> Dict:
        s = self._clean_state(state)
        msg = message.strip()
        low = msg.lower()

        if not s.get('topic'):
            s['topic'] = msg[:180]
            s['learning_goal'] = msg
            reply = f"Awesome! We will design your course around **{s['topic']}**.\n\nWhat is your current experience level: **Beginner**, **Intermediate**, or **Advanced**?"
        elif not s.get('skill_level'):
            if 'adv' in low:
                s['skill_level'] = 'Advanced'
            elif 'inter' in low:
                s['skill_level'] = 'Intermediate'
            else:
                s['skill_level'] = 'Beginner'
            reply = f"Set to **{s['skill_level']}**.\n\nHow many days would you like this course to span? (e.g. 14 days, 30 days, 60 days)"
        elif not s.get('duration_days') or s.get('duration_days') == 30 and 'day' in low:
            nums = re.findall(r'\d+', msg)
            if nums:
                val = int(nums[0])
                s['duration_days'] = val * 7 if 'week' in low else val
            reply = f"Target duration: **{s['duration_days']} days**.\n\nHow much study time can you commit each day? (e.g., 30 minutes, 1 hour)"
        elif not s.get('daily_minutes') or s.get('daily_minutes') == 60 and ('min' in low or 'hour' in low):
            nums = re.findall(r'\d+', msg)
            if nums:
                val = int(nums[0])
                s['daily_minutes'] = val * 60 if 'hour' in low else val
            reply = "Great schedule. What learning style do you prefer: **Practical exercises**, **Theory and concepts**, or a **Mixed** approach?"
        elif not s.get('learning_style'):
            if 'pract' in low:
                s['learning_style'] = 'Practical'
            elif 'theor' in low:
                s['learning_style'] = 'Theory'
            else:
                s['learning_style'] = 'Mixed'
            s['desired_outcome'] = f"Master {s['topic']} with practical projects and quizzes"
            summary = self._profile_summary(s)
            return {
                'state': s,
                'reply': f"{summary}\n\nYour profile is complete! Click **Generate Course** to build your curriculum.",
                'ready': True
            }
        else:
            summary = self._profile_summary(s)
            return {
                'state': s,
                'reply': f"{summary}\n\nProfile updated. You can click **Generate Course** whenever you're ready.",
                'ready': self._is_ready(s)
            }
        return {'state': s, 'reply': reply, 'ready': self._is_ready(s)}

    @staticmethod
    def _profile_summary(s: Dict) -> str:
        return (
            f"**Course Profile Summary**\n"
            f"- **Topic**: {s.get('topic','Not set')}\n"
            f"- **Level**: {s.get('skill_level','Beginner')}\n"
            f"- **Duration**: {s.get('duration_days', 30)} days ({s.get('daily_minutes', 60)} min/day)\n"
            f"- **Style**: {s.get('learning_style','Mixed')}\n"
            f"- **Language**: {s.get('language','English')}"
        )

    def _sample_course(self, p: Dict) -> Dict:
        """Deep, authentic topic-specific starter courses for key disciplines."""
        topic = (p.get('topic') or 'Python Programming').strip()
        low = topic.lower()
        level = p.get('skill_level') or 'Beginner'
        duration = int(p.get('duration_days') or 30)
        daily_min = int(p.get('daily_minutes') or 60)

        if any(k in low for k in ('python', 'py')):
            return self._build_python_course(level, duration, daily_min)
        elif any(k in low for k in ('machine learning', 'ml', 'ai', 'artificial intelligence')):
            return self._build_ml_course(level, duration, daily_min)
        elif any(k in low for k in ('sql', 'database', 'postgres', 'mysql')):
            return self._build_sql_course(level, duration, daily_min)
        elif any(k in low for k in ('web', 'html', 'javascript', 'react', 'full stack', 'frontend')):
            return self._build_web_course(level, duration, daily_min)
        elif any(k in low for k in ('java',)):
            return self._build_java_course(level, duration, daily_min)
        elif any(k in low for k in ('dsa', 'algorithm', 'data structure')):
            return self._build_dsa_course(level, duration, daily_min)
        else:
            return self._build_generic_course(topic, level, duration, daily_min)

    def _build_python_course(self, level: str, duration: int, daily_min: int) -> Dict:
        return {
            "title": f"Python Programming Masterclass: From Basics to Capstone ({level})",
            "description": f"A comprehensive {duration}-day specialization covering Python environment setup, fundamental syntax, control flow, data structures (lists, tuples, dicts, sets), modular functions, object-oriented programming, and file operations with automated testing.",
            "objectives": [
                "Write clean, idiomatic Python adhering to PEP 8 standards",
                "Manipulate complex data structures including lists, dictionaries, and sets",
                "Design modular functions with error handling and exception management",
                "Build and test an end-to-end data processing capstone project"
            ],
            "prerequisites": ["Basic computer literacy", "A modern web browser or Python 3.11 installed locally"],
            "capstone_project": {
                "title": "Automated Personal Finance & Expense Tracker",
                "description": "Develop a modular Python application that loads monthly transaction data from JSON/CSV files, calculates expenditure statistics by category, validates user entries, and generates summary reports.",
                "tasks": [
                    "Task 1: Design data models for transactions (date, category, amount, description)",
                    "Task 2: Implement file read/write with error handling for malformed data",
                    "Task 3: Build calculation functions for totals, category averages, and monthly trends",
                    "Task 4: Add input validation and unit tests verifying calculation correctness"
                ],
                "starter_code": "def process_expenses(transactions):\n    \"\"\"Calculate total and category breakdown.\"\"\"\n    summary = {'total': 0, 'by_category': {}}\n    for t in transactions:\n        summary['total'] += t['amount']\n        cat = t['category']\n        summary['by_category'][cat] = summary['by_category'].get(cat, 0) + t['amount']\n    return summary\n",
                "deliverables": "Working script, sample transaction datasets, and a verification test suite."
            },
            "sources": [
                {"title": "Official Python 3 Documentation", "url": "https://docs.python.org/3/"},
                {"title": "PEP 8 Style Guide for Python Code", "url": "https://peps.python.org/pep-0008/"}
            ],
            "modules": [
                {
                    "title": "Module 1: Environment Setup, Variables & Expressions",
                    "description": "Master the Python interpreter, variable binding, primitive types, operators, and dynamic typing mechanics.",
                    "duration_minutes": 180,
                    "objectives": ["Set up Python environments", "Understand numeric and string types", "Perform arithmetic and string operations"],
                    "lessons": [
                        {
                            "title": "Lesson 1: Python Architecture & Variables",
                            "explanation": "Python is an interpreted, high-level, dynamically typed language. When you execute Python code, source code is compiled into bytecode (.pyc) and executed on the Python Virtual Machine (PVM). Variable names in Python are references to objects stored in memory rather than labeled memory slots. This means variables can change types dynamically during execution.",
                            "concepts": ["Interpreted vs compiled execution", "Dynamic typing and type inference", "Memory references and garbage collection"],
                            "examples": [
                                "```python\n# Variable assignment and type inspection\nitem_name = 'CourseMind AI'\nprice = 49.99\nquantity = 2\nis_active = True\ntotal = price * quantity\nprint(f'{item_name} Total: ${total:.2f}')\nprint('Type of total:', type(total))\n```\nExpected Output:\n```\nCourseMind AI Total: $99.98\nType of total: <class 'float'>\n```"
                            ],
                            "exercises": ["Declare variables for learner name, age, and enrolled courses", "Calculate the average study time across 5 days"],
                            "takeaways": ["Variables are object references in Python", "Use f-strings for readable formatted output"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function calculate_total(price, quantity, discount_pct) that calculates the discounted total.",
                                "starter_code": "def calculate_total(price, quantity, discount_pct):\n    # Calculate subtotal, apply discount, return final float\n    pass\n",
                                "solution": "def calculate_total(price, quantity, discount_pct):\n    subtotal = price * quantity\n    return round(subtotal * (1 - discount_pct / 100), 2)\n",
                                "test_cases": [
                                    {"input": "(100, 2, 10)", "expected": "180.0"},
                                    {"input": "(50, 1, 0)", "expected": "50.0"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Primitive Data Types & String Manipulation",
                            "explanation": "Python features robust built-in string and numeric types. Strings in Python are immutable sequences of Unicode characters, meaning methods like .upper() or .replace() return new strings rather than mutating the original. Slicing with [start:stop:step] provides flexible sequence extraction.",
                            "concepts": ["String immutability and memory efficiency", "String methods (strip, split, join, replace)", "Slice syntax: [start:stop:step]"],
                            "examples": [
                                "```python\ntext = '  welcome to python learning  '\ncleaned = text.strip().title()\nwords = cleaned.split()\nslug = '-'.join(words)\nprint('Original:', repr(text))\nprint('Slug:', slug)\n```\nExpected Output:\n```\nOriginal: '  welcome to python learning  '\nSlug: Welcome-To-Python-Learning\n```"
                            ],
                            "exercises": ["Extract the username and domain from an email address", "Reverse a string using step slicing [::-1]"],
                            "takeaways": ["Strings are immutable sequences", "Use .join() to concatenate collections of strings efficiently"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function clean_text(raw_str) that trims whitespace, converts to title case, and replaces spaces with underscores.",
                                "starter_code": "def clean_text(raw_str):\n    # Return cleaned and formatted string\n    pass\n",
                                "solution": "def clean_text(raw_str):\n    return raw_str.strip().title().replace(' ', '_')\n",
                                "test_cases": [
                                    {"input": "'  hello python world  '", "expected": "'Hello_Python_World'"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Numeric Operations & Type Conversions",
                            "explanation": "Python distinguishes between integers of arbitrary precision and IEEE 754 floating-point numbers. Floor division (//) returns the quotient without remainder, while modulus (%) returns the remainder. Explicit type conversions with int(), float(), and str() prevent runtime TypeErrors.",
                            "concepts": ["Integer arbitrary precision vs float representation", "Floor division (//) and modulo (%)", "Typecasting functions and ValueError handling"],
                            "examples": [
                                "```python\ntotal_minutes = 245\nhours = total_minutes // 60\nremaining_minutes = total_minutes % 60\nprint(f'{total_minutes} minutes = {hours} hours and {remaining_minutes} minutes')\n```\nExpected Output:\n```\n245 minutes = 4 hours and 5 minutes\n```"
                            ],
                            "exercises": ["Convert Celsius to Fahrenheit using (C * 9/5) + 32", "Check whether a given integer is even or odd using % 2"],
                            "takeaways": ["Use // for integer division and % for modulo", "Beware floating point rounding nuances in financial math"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function convert_time(total_seconds) that returns a dictionary {'hours': H, 'minutes': M, 'seconds': S}.",
                                "starter_code": "def convert_time(total_seconds):\n    # Return dict with hours, minutes, seconds\n    pass\n",
                                "solution": "def convert_time(total_seconds):\n    h = total_seconds // 3600\n    m = (total_seconds % 3600) // 60\n    s = total_seconds % 60\n    return {'hours': h, 'minutes': m, 'seconds': s}\n",
                                "test_cases": [
                                    {"input": "3665", "expected": "{'hours': 1, 'minutes': 1, 'seconds': 5}"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What happens when you reassign a variable to a different type in Python (e.g., x = 5 then x = 'hello')?",
                            "options": [
                                "The variable name is rebound to a new string object in memory",
                                "A compilation TypeError occurs immediately",
                                "The integer 5 is converted in place into an ASCII string",
                                "Python creates a duplicate variable under a shadow pointer"
                            ],
                            "correct_index": 0,
                            "explanation": "Python is dynamically typed. Variable names are bindings to heap-allocated objects; rebinding points the identifier to a new object."
                        },
                        {
                            "prompt": "Which expression extracts the substring 'Python' from 'Learn Python Today' using slice notation?",
                            "options": [
                                "'Learn Python Today'[6:12]",
                                "'Learn Python Today'[5:11]",
                                "'Learn Python Today'[6:11]",
                                "'Learn Python Today'[7:13]"
                            ],
                            "correct_index": 0,
                            "explanation": "Index 6 corresponds to 'P' and index 12 is the non-inclusive stop boundary after 'n'."
                        },
                        {
                            "prompt": "What is the result of the expression 17 // 4 and 17 % 4?",
                            "options": [
                                "4 and 1",
                                "4.25 and 1",
                                "4 and 0.25",
                                "5 and 1"
                            ],
                            "correct_index": 0,
                            "explanation": "Floor division // yields 4 (since 4*4=16) and modulo % yields the remainder 1."
                        },
                        {
                            "prompt": "Why is 'hello'.upper() not modifying the original string in place?",
                            "options": [
                                "Strings in Python are immutable sequences",
                                "Methods starting with lowercase letters cannot alter memory",
                                "Strings require explicit pointer references to change",
                                "In-place string modification is only supported inside functions"
                            ],
                            "correct_index": 0,
                            "explanation": "All string operations in Python return a new string because str objects are immutable."
                        }
                    ]
                },
                {
                    "title": "Module 2: Control Flow & Conditional Branching",
                    "description": "Structure program execution with if-elif-else statements, boolean logic, ternary operators, and comparison expressions.",
                    "duration_minutes": 160,
                    "objectives": ["Implement complex conditional trees", "Apply short-circuit boolean evaluation", "Use the ternary conditional expression"],
                    "lessons": [
                        {
                            "title": "Lesson 1: Conditional Logic & Boolean Operators",
                            "explanation": "Decision making in Python relies on if, elif, and else statements evaluated with boolean expressions. Truth value testing evaluates non-zero numbers and non-empty collections as True, while 0, None, '', [], and {} evaluate as False. The logical operators 'and', 'or', and 'not' evaluate using short-circuit logic.",
                            "concepts": ["Boolean evaluation & truthy/falsy values", "Short-circuit evaluation in and/or", "Comparison operators (==, !=, <, >, in, is)"],
                            "examples": [
                                "```python\nscore = 88\nif score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelif score >= 70:\n    grade = 'C'\nelse:\n    grade = 'Needs Improvement'\nstatus = 'Passed' if score >= 70 else 'Retake'\nprint(f'Score: {score} -> Grade: {grade} ({status})')\n```\nExpected Output:\n```\nScore: 88 -> Grade: B (Passed)\n```"
                            ],
                            "exercises": ["Write a leap year checker using modulo conditions", "Evaluate user access levels based on role and active status"],
                            "takeaways": ["Colons and consistent indentation define conditional blocks", "Prefer 'is None' over '== None' for identity checks"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function categorize_age(age) returning 'Child' (0-12), 'Teen' (13-19), 'Adult' (20-64), or 'Senior' (65+).",
                                "starter_code": "def categorize_age(age):\n    pass\n",
                                "solution": "def categorize_age(age):\n    if age <= 12:\n        return 'Child'\n    elif age <= 19:\n        return 'Teen'\n    elif age <= 64:\n        return 'Adult'\n    return 'Senior'\n",
                                "test_cases": [
                                    {"input": "15", "expected": "'Teen'"},
                                    {"input": "30", "expected": "'Adult'"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Loops & Iteration Mechanics (for and while)",
                            "explanation": "Python offers 'for' loops for iterating over sequences (iterables) and 'while' loops for repeating actions until a condition becomes false. The range() generator provides memory-efficient numerical sequences. The loop controls 'break' (exit loop early) and 'continue' (skip to next iteration) govern fine-grained loop flow.",
                            "concepts": ["for loop over iterables vs range()", "while loop condition management and avoiding infinite loops", "break, continue, and the loop-else construct"],
                            "examples": [
                                "```python\nprimes = []\nfor num in range(2, 20):\n    is_prime = True\n    for divisor in range(2, int(num ** 0.5) + 1):\n        if num % divisor == 0:\n            is_prime = False\n            break\n    if is_prime:\n        primes.append(num)\nprint('Primes under 20:', primes)\n```\nExpected Output:\n```\nPrimes under 20: [2, 3, 5, 7, 11, 13, 17, 19]\n```"
                            ],
                            "exercises": ["Calculate the factorial of N using a while loop", "Find the sum of all multiples of 3 or 5 under 100"],
                            "takeaways": ["range(start, stop, step) generates values on-the-fly without allocating memory", "Use enumerate() when loop index is needed alongside items"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function sum_evens(numbers) that returns the sum of all even integers in the given list.",
                                "starter_code": "def sum_evens(numbers):\n    pass\n",
                                "solution": "def sum_evens(numbers):\n    return sum(x for x in numbers if x % 2 == 0)\n",
                                "test_cases": [
                                    {"input": "[1, 2, 3, 4, 5, 6]", "expected": "12"},
                                    {"input": "[1, 3, 5]", "expected": "0"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Iteration Patterns with enumerate and zip",
                            "explanation": "Idiomatic Python avoids C-style index counters like 'for i in range(len(items))'. Instead, enumerate(items) yields (index, item) tuples natively. Similarly, zip(list_a, list_b) pairs corresponding elements from multiple iterables simultaneously, terminating when the shortest sequence is exhausted.",
                            "concepts": ["enumerate() for index-value pairs", "zip() for multi-sequence lockstep iteration", "Unpacking tuples in loop headers"],
                            "examples": [
                                "```python\nstudents = ['Alice', 'Bob', 'Charlie']\nscores = [94, 87, 91]\nfor rank, (name, score) in enumerate(zip(students, scores), start=1):\n    print(f'{rank}. {name}: {score}%')\n```\nExpected Output:\n```\n1. Alice: 94%\n2. Bob: 87%\n3. Charlie: 91%\n```"
                            ],
                            "exercises": ["Pair product names with prices and apply discount calculations", "Format an indexed table of completed lesson titles"],
                            "takeaways": ["Avoid range(len(...)) — use enumerate() instead", "zip(*matrix) can transpose row-oriented data into columns"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function create_roster(names, grades) returning a list of strings 'Name: Grade'.",
                                "starter_code": "def create_roster(names, grades):\n    pass\n",
                                "solution": "def create_roster(names, grades):\n    return [f'{n}: {g}' for n, g in zip(names, grades)]\n",
                                "test_cases": [
                                    {"input": "(['Alice', 'Bob'], ['A', 'B'])", "expected": "['Alice: A', 'Bob: B']"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What does Python evaluate an empty list [] as inside an if condition?",
                            "options": [
                                "Falsy (evaluates to False)",
                                "Truthy (evaluates to True)",
                                "Raises an IndexError exception",
                                "Evaluates to None"
                            ],
                            "correct_index": 0,
                            "explanation": "Empty collections (lists, dicts, tuples, sets, strings) evaluate as False in boolean contexts."
                        },
                        {
                            "prompt": "How does the 'continue' keyword modify loop execution?",
                            "options": [
                                "Skips the remainder of the current iteration and advances to the next",
                                "Terminates the loop immediately",
                                "Restarts the loop from index 0",
                                "Skips the enclosing function"
                            ],
                            "correct_index": 0,
                            "explanation": "continue halts the current loop cycle and advances directly to the next iteration."
                        },
                        {
                            "prompt": "Which function pairs two lists ['a', 'b'] and [1, 2] into [('a', 1), ('b', 2)]?",
                            "options": [
                                "zip(['a', 'b'], [1, 2])",
                                "pair(['a', 'b'], [1, 2])",
                                "join(['a', 'b'], [1, 2])",
                                "merge(['a', 'b'], [1, 2])"
                            ],
                            "correct_index": 0,
                            "explanation": "zip takes iterables and aggregates corresponding elements into tuples."
                        },
                        {
                            "prompt": "What is the primary benefit of enumerate(items, start=1)?",
                            "options": [
                                "Provides both index and item without manual counter variables",
                                "Reverses the order of items automatically",
                                "Enforces sorted alphabetical ordering",
                                "Converts all items to integers"
                            ],
                            "correct_index": 0,
                            "explanation": "enumerate cleanly yields (index, value) tuples without needing a manual loop counter."
                        }
                    ]
                },
                {
                    "title": "Module 3: Core Data Structures (Lists, Tuples, Dictionaries, Sets)",
                    "description": "Master Python's built-in collections, mutability trade-offs, hashing mechanics, and list/dict comprehensions.",
                    "duration_minutes": 200,
                    "objectives": ["Choose the right collection type for specific time-complexity needs", "Master list and dictionary comprehensions", "Understand hashable keys and dictionary lookup O(1) performance"],
                    "lessons": [
                        {
                            "title": "Lesson 1: Lists & Dynamic Arrays",
                            "explanation": "Lists are mutable, ordered sequences implemented as dynamic arrays of pointers in CPython. Appending to a list is amortized O(1), whereas inserting or deleting at index 0 requires shifting all subsequent elements, taking O(n) time. List comprehensions provide clean, concise syntax for mapping and filtering.",
                            "concepts": ["List mutability and in-place methods (.append, .extend, .pop)", "Amortized O(1) append vs O(n) insert", "List comprehensions with conditional filters"],
                            "examples": [
                                "```python\nnumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nsquares_of_evens = [x ** 2 for x in numbers if x % 2 == 0]\nprint('Squares of evens:', squares_of_evens)\n```\nExpected Output:\n```\nSquares of evens: [4, 16, 36, 64, 100]\n```"
                            ],
                            "exercises": ["Filter a list of prices to remove outliers", "Flatten a 2D matrix into a 1D list using comprehension"],
                            "takeaways": ["Use list comprehensions instead of map() and filter() in modern Python", "Lists are ordered and allow duplicate elements"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function filter_positive_squares(nums) that returns the square of each strictly positive number.",
                                "starter_code": "def filter_positive_squares(nums):\n    pass\n",
                                "solution": "def filter_positive_squares(nums):\n    return [x * x for x in nums if x > 0]\n",
                                "test_cases": [
                                    {"input": "[-2, -1, 0, 3, 4]", "expected": "[9, 16]"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Dictionaries & Hash Maps",
                            "explanation": "Dictionaries in Python are ordered hash tables mapping hashable keys to arbitrary values. Keys must be immutable (strings, numbers, tuples without mutable items). Looking up, adding, or deleting a key runs in average O(1) time. The .get(key, default) method prevents KeyErrors when retrieving absent keys.",
                            "concepts": ["Hash tables and O(1) average lookup", "Hashable key requirements", "dict methods: .get(), .items(), .keys(), .values()"],
                            "examples": [
                                "```python\ncourse_stats = {'python': 450, 'java': 320, 'sql': 280}\ncourse_stats['web_dev'] = 390\nprint('Python learners:', course_stats.get('python', 0))\nprint('Ruby learners:', course_stats.get('ruby', 0))\nfor course, count in sorted(course_stats.items(), key=lambda x: x[1], reverse=True):\n    print(f'- {course}: {count}')\n```\nExpected Output:\n```\nPython learners: 450\nRuby learners: 0\n- python: 450\n- web_dev: 390\n- java: 320\n- sql: 280\n```"
                            ],
                            "exercises": ["Count word frequency in a paragraph using a dictionary", "Invert a dictionary to map values to lists of keys"],
                            "takeaways": ["Dictionaries maintain insertion order since Python 3.7", "Use dict.get(key, default) to safely avoid KeyError exceptions"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function word_frequency(words) that returns a dict mapping each word to its occurrence count.",
                                "starter_code": "def word_frequency(words):\n    pass\n",
                                "solution": "def word_frequency(words):\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n",
                                "test_cases": [
                                    {"input": "['apple', 'banana', 'apple']", "expected": "{'apple': 2, 'banana': 1}"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Tuples & Sets",
                            "explanation": "Tuples are immutable ordered sequences, ideal for fixed records and dictionary keys. Sets are mutable, unordered collections of unique hashable elements implemented with hash tables. Set mathematical operations like union (|), intersection (&), and difference (-) run in O(len(set)) time.",
                            "concepts": ["Tuples as immutable records", "Set uniqueness and fast O(1) membership checks", "Set algebra: union (|), intersection (&), difference (-)"],
                            "examples": [
                                "```python\npython_students = {'Alice', 'Bob', 'Charlie', 'Diana'}\nsql_students = {'Bob', 'Diana', 'Edward', 'Frank'}\nboth = python_students & sql_students\neither = python_students | sql_students\nonly_python = python_students - sql_students\nprint('Both courses:', sorted(both))\nprint('Only Python:', sorted(only_python))\n```\nExpected Output:\n```\nBoth courses: ['Bob', 'Diana']\nOnly Python: ['Alice', 'Charlie']\n```"
                            ],
                            "exercises": ["Remove duplicate values from a list while maintaining set logic", "Check whether one set is a subset of another using <="],
                            "takeaways": ["Tuples cannot be altered after creation", "Checking 'item in set' is O(1) vs O(n) for lists"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function find_common(list1, list2) that returns a sorted list of unique elements present in both lists.",
                                "starter_code": "def find_common(list1, list2):\n    pass\n",
                                "solution": "def find_common(list1, list2):\n    return sorted(list(set(list1) & set(list2)))\n",
                                "test_cases": [
                                    {"input": "([1, 2, 3], [2, 3, 4])", "expected": "[2, 3]"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What is the average time complexity of looking up a key in a Python dictionary?",
                            "options": [
                                "O(1) constant time",
                                "O(n) linear time",
                                "O(log n) logarithmic time",
                                "O(n^2) quadratic time"
                            ],
                            "correct_index": 0,
                            "explanation": "Python dictionaries use hash tables, providing average O(1) key lookups, insertions, and deletions."
                        },
                        {
                            "prompt": "Why can a Python list NOT be used as a key in a standard dictionary?",
                            "options": [
                                "Lists are mutable and do not implement __hash__",
                                "Lists cannot be converted to strings",
                                "Dictionary keys can only be integer primitives",
                                "Lists consume too much memory for hash tables"
                            ],
                            "correct_index": 0,
                            "explanation": "Dictionary keys must be hashable and immutable. Because lists are mutable, their contents and hash value could change."
                        },
                        {
                            "prompt": "Which set operation finds elements that exist in set_a but NOT in set_b?",
                            "options": [
                                "set_a - set_b (difference)",
                                "set_a & set_b (intersection)",
                                "set_a | set_b (union)",
                                "set_a ^ set_b (symmetric difference)"
                            ],
                            "correct_index": 0,
                            "explanation": "The minus operator (-) calculates set difference, returning elements present exclusively in the left set."
                        },
                        {
                            "prompt": "What does my_dict.get('missing_key', 0) return if 'missing_key' is absent?",
                            "options": [
                                "0",
                                "Raises KeyError",
                                "None",
                                "Empty string ''"
                            ],
                            "correct_index": 0,
                            "explanation": "dict.get(key, default) safely returns the supplied default value if the key does not exist."
                        }
                    ]
                },
                {
                    "title": "Module 4: Functions, Scope & Functional Paradigms",
                    "description": "Design modular, reusable functions with positional arguments, keyword arguments, *args, **kwargs, lambda functions, and closures.",
                    "duration_minutes": 180,
                    "objectives": ["Write modular functions with type hints and docstrings", "Handle variable positional and keyword arguments (*args, **kwargs)", "Apply lambda functions and higher-order functions"],
                    "lessons": [
                        {
                            "title": "Lesson 1: Function Definition, Parameters & Return Values",
                            "explanation": "Functions in Python are defined with the 'def' keyword and are first-class citizens: they can be passed as arguments, assigned to variables, and returned from other functions. Default parameter values are evaluated once at function definition time (avoid mutable default arguments like def f(x=[])).",
                            "concepts": ["def statement, return values, None default return", "Positional vs keyword arguments", "Mutable default argument pitfall"],
                            "examples": [
                                "```python\ndef calculate_discount(price: float, discount: float = 0.1) -> float:\n    \"\"\"Calculate price after discount percentage.\"\"\"\n    return round(price * (1 - discount), 2)\n\nprint('Standard:', calculate_discount(100.0))\nprint('Custom:', calculate_discount(100.0, discount=0.25))\n```\nExpected Output:\n```\nStandard: 90.0\nCustom: 75.0\n```"
                            ],
                            "exercises": ["Write a function calculating compound interest", "Build a validator checking string length and character constraints"],
                            "takeaways": ["Always use None as default for mutable parameters like lists or dicts", "Type hints and docstrings document intent clearly"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function append_item(val, target_list=None) that safely appends val to target_list (creating a new list if None).",
                                "starter_code": "def append_item(val, target_list=None):\n    pass\n",
                                "solution": "def append_item(val, target_list=None):\n    if target_list is None:\n        target_list = []\n    target_list.append(val)\n    return target_list\n",
                                "test_cases": [
                                    {"input": "(5, None)", "expected": "[5]"},
                                    {"input": "(3, [1, 2])", "expected": "[1, 2, 3]"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Variable Arguments (*args & **kwargs)",
                            "explanation": "The *args parameter packs surplus positional arguments into a tuple, while **kwargs packs surplus keyword arguments into a dictionary. This allows functions to accept arbitrary parameters, common when writing decorators, API wrappers, and factory methods.",
                            "concepts": ["*args for positional argument packing", "**kwargs for keyword argument packing", "Argument unpacking operator (*)"],
                            "examples": [
                                "```python\ndef format_log(level: str, *messages, **metadata):\n    body = ' '.join(str(m) for m in messages)\n    tags = ', '.join(f'{k}={v}' for k, v in metadata.items())\n    print(f'[{level.upper()}] {body} | {tags}')\n\nformat_log('info', 'User', 'logged in', user_id=42, ip='127.0.0.1')\n```\nExpected Output:\n```\n[INFO] User logged in | user_id=42, ip=127.0.0.1\n```"
                            ],
                            "exercises": ["Write a multiply_all(*numbers) function", "Build a configurable URL query builder using **kwargs"],
                            "takeaways": ["*args creates a tuple; **kwargs creates a dictionary", "Use unpacking (*list, **dict) when passing arguments dynamically"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function multiply_all(*args) that multiplies all given numbers together (returns 1 if empty).",
                                "starter_code": "def multiply_all(*args):\n    pass\n",
                                "solution": "def multiply_all(*args):\n    total = 1\n    for n in args:\n        total *= n\n    return total\n",
                                "test_cases": [
                                    {"input": "(2, 3, 4)", "expected": "24"},
                                    {"input": "()", "expected": "1"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Variable Scope (LEGB) & Lambda Functions",
                            "explanation": "Python resolves variable identifiers following the LEGB rule: Local, Enclosing (nested functions), Global (module level), and Built-in. Lambda functions provide anonymous, inline expressions restricted to a single expression, commonly used as key extractors in sorted() and min().",
                            "concepts": ["LEGB scope resolution rule", "The 'global' and 'nonlocal' keywords", "Anonymous lambda functions: lambda x: x + 1"],
                            "examples": [
                                "```python\nproducts = [\n    {'name': 'Laptop', 'price': 999, 'rating': 4.8},\n    {'name': 'Mouse', 'price': 25, 'rating': 4.2},\n    {'name': 'Monitor', 'price': 220, 'rating': 4.7}\n]\n# Sort by price ascending\nby_price = sorted(products, key=lambda p: p['price'])\n# Sort by rating descending\nby_rating = sorted(products, key=lambda p: p['rating'], reverse=True)\nprint('Cheapest:', by_price[0]['name'])\nprint('Highest rated:', by_rating[0]['name'])\n```\nExpected Output:\n```\nCheapest: Mouse\nHighest rated: Laptop\n```"
                            ],
                            "exercises": ["Sort strings by length using key=lambda s: len(s)", "Create a counter closure using nonlocal variable binding"],
                            "takeaways": ["LEGB determines where Python looks for variable bindings", "Use lambda functions for small transformations, not complex logic"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function sort_by_length(words) that returns the list sorted by length ascending using a lambda.",
                                "starter_code": "def sort_by_length(words):\n    pass\n",
                                "solution": "def sort_by_length(words):\n    return sorted(words, key=lambda w: len(w))\n",
                                "test_cases": [
                                    {"input": "['apple', 'pie', 'banana']", "expected": "['pie', 'apple', 'banana']"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What danger occurs when defining 'def add_item(item, items=[])'?",
                            "options": [
                                "The default list is created once at definition and shared across all calls",
                                "Python throws a SyntaxError on mutable arguments",
                                "The list is deleted when the function exits",
                                "Python converts the list into an immutable tuple"
                            ],
                            "correct_index": 0,
                            "explanation": "Default parameter values are evaluated once at module load time. Reusing a mutable default shares state across calls."
                        },
                        {
                            "prompt": "What does the LEGB acronym in Python scoping stand for?",
                            "options": [
                                "Local, Enclosing, Global, Built-in",
                                "Logical, External, Global, Bytecode",
                                "Lexical, Environment, Garbage, Binary",
                                "Literal, Evaluated, Grouped, Bound"
                            ],
                            "correct_index": 0,
                            "explanation": "LEGB describes the resolution order: Local scope, Enclosing closures, Global module scope, and Built-ins."
                        },
                        {
                            "prompt": "What type does *args arrive as inside a function definition?",
                            "options": [
                                "A tuple",
                                "A list",
                                "A set",
                                "A dictionary"
                            ],
                            "correct_index": 0,
                            "explanation": "Excess positional arguments gathered by *args are bundled into an immutable tuple."
                        },
                        {
                            "prompt": "Which expression uses a lambda to sort tuples by their second element?",
                            "options": [
                                "sorted(pairs, key=lambda x: x[1])",
                                "sorted(pairs, order=x[1])",
                                "pairs.sort(by=1)",
                                "sorted(pairs, lambda: 1)"
                            ],
                            "correct_index": 0,
                            "explanation": "The key argument expects a callable. lambda x: x[1] extracts the second element of each tuple for sorting."
                        }
                    ]
                },
                {
                    "title": "Module 5: Object-Oriented Programming (OOP) & Exception Handling",
                    "description": "Construct robust classes with encapsulation, inheritance, dunder methods (__init__, __str__), and manage errors with try-except-finally blocks.",
                    "duration_minutes": 220,
                    "objectives": ["Design classes with constructors and instance methods", "Apply inheritance and polymorphism", "Catch and raise specific exceptions safely"],
                    "lessons": [
                        {
                            "title": "Lesson 1: Classes, Objects & The __init__ Constructor",
                            "explanation": "Object-Oriented Programming models systems as interacting objects possessing state (attributes) and behavior (methods). A class is a blueprint; an instance is an allocated object. In Python, methods take 'self' as their first parameter, referencing the instance. The __init__ method initializes attributes upon instantiation.",
                            "concepts": ["Classes as blueprints, objects as instances", "The 'self' instance pointer", "The __init__ constructor and __repr__ / __str__ dunder methods"],
                            "examples": [
                                "```python\nclass Student:\n    def __init__(self, name: str, student_id: int):\n        self.name = name\n        self.student_id = student_id\n        self.grades = []\n    \n    def add_grade(self, grade: float):\n        self.grades.append(grade)\n        \n    def gpa(self) -> float:\n        return round(sum(self.grades) / len(self.grades), 2) if self.grades else 0.0\n\ns = Student('Alex', 101)\ns.add_grade(85.0)\ns.add_grade(92.0)\nprint(f'{s.name} GPA:', s.gpa())\n```\nExpected Output:\n```\nAlex GPA: 88.5\n```"
                            ],
                            "exercises": ["Design a BankAccount class with deposit, withdraw, and balance methods", "Implement an InventoryItem class with discount calculation"],
                            "takeaways": ["self binds method calls to specific instance state", "Implement __str__ and __repr__ to aid debugging and logging"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Implement a Rectangle class with width and height, and methods area() and perimeter().",
                                "starter_code": "class Rectangle:\n    def __init__(self, width, height):\n        pass\n    def area(self):\n        pass\n    def perimeter(self):\n        pass\n",
                                "solution": "class Rectangle:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height\n    def area(self):\n        return self.width * self.height\n    def perimeter(self):\n        return 2 * (self.width + self.height)\n",
                                "test_cases": [
                                    {"input": "Rectangle(4, 5).area()", "expected": "20"},
                                    {"input": "Rectangle(4, 5).perimeter()", "expected": "18"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Inheritance & Polymorphism",
                            "explanation": "Inheritance allows child classes to reuse and extend parent functionality. The super() function delegates calls to parent classes. Polymorphism enables uniform interfaces across different object types, allowing functions to interact with any object providing the expected methods (duck typing).",
                            "concepts": ["Class inheritance syntax: class Child(Parent)", "super().__init__() method chaining", "Polymorphism and duck typing in Python"],
                            "examples": [
                                "```python\nclass Vehicle:\n    def __init__(self, brand: str):\n        self.brand = brand\n    def description(self) -> str:\n        return f'Vehicle from {self.brand}'\n\nclass ElectricCar(Vehicle):\n    def __init__(self, brand: str, battery_kwh: int):\n        super().__init__(brand)\n        self.battery_kwh = battery_kwh\n    def description(self) -> str:\n        return f'{self.brand} Electric Car ({self.battery_kwh} kWh)'\n\ncar = ElectricCar('Tesla', 75)\nprint(car.description())\n```\nExpected Output:\n```\nTesla Electric Car (75 kWh)\n```"
                            ],
                            "exercises": ["Build an Employee and Manager hierarchy calculating bonuses", "Implement a Shape base class with Circle and Square subclasses"],
                            "takeaways": ["Use super() to initialize parent class state cleanly", "Favor composition over inheritance when relationships are 'has-a' rather than 'is-a'"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Create a Square class inheriting from Rectangle that initializes width and height to the same side length.",
                                "starter_code": "class Rectangle:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height\n    def area(self):\n        return self.width * self.height\n\nclass Square(Rectangle):\n    pass\n",
                                "solution": "class Square(Rectangle):\n    def __init__(self, side):\n        super().__init__(side, side)\n",
                                "test_cases": [
                                    {"input": "Square(5).area()", "expected": "25"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Robust Exception Handling (try-except-else-finally)",
                            "explanation": "Runtime errors in Python raise exceptions. Handling errors with try-except blocks ensures applications degrade gracefully rather than crashing. Catch specific exceptions (ValueError, FileNotFoundError) rather than catching bare Exception. The 'else' block executes if no error was raised, and 'finally' always runs.",
                            "concepts": ["try, except, else, finally structure", "Catching specific exceptions vs bare except", "Raising exceptions with raise and custom Exception classes"],
                            "examples": [
                                "```python\ndef safe_divide(a: float, b: float) -> Optional[float]:\n    try:\n        result = a / b\n    except ZeroDivisionError:\n        print('Error: Division by zero!')\n        return None\n    except TypeError as e:\n        print(f'Error: Invalid types ({e})')\n        return None\n    else:\n        return round(result, 3)\n    finally:\n        # Clean up resources\n        pass\n\nprint('Result 1:', safe_divide(10, 2))\nprint('Result 2:', safe_divide(10, 0))\n```\nExpected Output:\n```\nResult 1: 5.0\nError: Division by zero!\nResult 2: None\n```"
                            ],
                            "exercises": ["Validate user integer inputs with try-except ValueError", "Write a file reading routine with FileNotFoundError handling"],
                            "takeaways": ["Never use bare except: pass; it swallows SystemExit and KeyboardInterrupt", "Use the 'else' block for code that should only run if try succeeded"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function parse_int(val, default_val) that converts val to int, returning default_val if ValueError or TypeError occurs.",
                                "starter_code": "def parse_int(val, default_val):\n    pass\n",
                                "solution": "def parse_int(val, default_val):\n    try:\n        return int(val)\n    except (ValueError, TypeError):\n        return default_val\n",
                                "test_cases": [
                                    {"input": "('42', 0)", "expected": "42"},
                                    {"input": "('abc', 0)", "expected": "0"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What is the purpose of the first parameter 'self' in Python class methods?",
                            "options": [
                                "It points to the specific instance of the class executing the method",
                                "It declares the method as static",
                                "It initializes the class in global memory",
                                "It creates a private variable namespace"
                            ],
                            "correct_index": 0,
                            "explanation": "When you invoke obj.method(), Python automatically passes obj as the first argument, conventionally named self."
                        },
                        {
                            "prompt": "When does the 'finally' block execute in a try-except statement?",
                            "options": [
                                "Always, whether an exception occurred or not",
                                "Only when an exception was raised and caught",
                                "Only when no exception was raised",
                                "Only if a return statement was executed"
                            ],
                            "correct_index": 0,
                            "explanation": "The finally block executes unconditionally upon leaving the try/except structure, making it ideal for cleanup."
                        },
                        {
                            "prompt": "How does a child class invoke its parent constructor in Python 3?",
                            "options": [
                                "super().__init__(...)",
                                "parent.__init__(self, ...)",
                                "base.construct(...)",
                                "this.super(...)"
                            ],
                            "correct_index": 0,
                            "explanation": "super() delegates method resolution to parent classes in accordance with Method Resolution Order (MRO)."
                        },
                        {
                            "prompt": "Why should catching bare 'except:' be avoided in production code?",
                            "options": [
                                "It catches KeyboardInterrupt and SystemExit, preventing graceful program termination",
                                "It runs slower than checking types manually",
                                "It causes memory leaks in CPython",
                                "It is deprecated and forbidden in Python 3"
                            ],
                            "correct_index": 0,
                            "explanation": "Bare 'except:' catches BaseException, which includes user exit signals like Ctrl+C (KeyboardInterrupt)."
                        }
                    ]
                },
                {
                    "title": "Module 6: File I/O, Modules & Capstone Project",
                    "description": "Read and write text and JSON files using context managers ('with'), import standard library modules, and integrate all learned concepts into a complete capstone project.",
                    "duration_minutes": 240,
                    "objectives": ["Safely read and write files with context managers", "Serialize and deserialize JSON data", "Build and test the comprehensive capstone application"],
                    "lessons": [
                        {
                            "title": "Lesson 1: File Operations & Context Managers (with statement)",
                            "explanation": "Accessing files on disk requires opening a file handle and closing it when complete. The 'with open(...) as f:' context manager guarantees the file descriptor is closed automatically, even if an exception occurs during reading or writing. Modes include 'r' (read), 'w' (write/truncate), 'a' (append), and 'r+' (read/write).",
                            "concepts": ["The with statement and the context manager protocol", "File modes ('r', 'w', 'a', 'b')", "Iterating over file handles line-by-line"],
                            "examples": [
                                "```python\n# Safe file writing and reading\ndata = ['User: Alice, Score: 95\\n', 'User: Bob, Score: 88\\n']\n# Context manager automatically closes file handle\nwith open('scores.txt', 'w') as f:\n    f.writelines(data)\n\nwith open('scores.txt', 'r') as f:\n    for line in f:\n        print('Read line:', line.strip())\n```\nExpected Output:\n```\nRead line: User: Alice, Score: 95\nRead line: User: Bob, Score: 88\n```"
                            ],
                            "exercises": ["Count word occurrences across an entire text file", "Filter log files for lines containing 'ERROR'"],
                            "takeaways": ["Always use 'with open()' to guarantee file handles close properly", "Use 'with open(..., encoding=\"utf-8\")' to avoid platform encoding bugs"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function parse_csv_lines(csv_text) that takes a multi-line CSV string and returns a list of dictionaries using the first row as headers.",
                                "starter_code": "def parse_csv_lines(csv_text):\n    pass\n",
                                "solution": "def parse_csv_lines(csv_text):\n    lines = [l.strip() for l in csv_text.strip().split('\\n') if l.strip()]\n    if not lines:\n        return []\n    headers = [h.strip() for h in lines[0].split(',')]\n    rows = []\n    for line in lines[1:]:\n        vals = [v.strip() for v in line.split(',')]\n        rows.append(dict(zip(headers, vals)))\n    return rows\n",
                                "test_cases": [
                                    {"input": "'name,age\\nAlice,25\\nBob,30'", "expected": "[{'name': 'Alice', 'age': '25'}, {'name': 'Bob', 'age': '30'}]"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: JSON Serialization & The Standard Library",
                            "explanation": "JSON (JavaScript Object Notation) is the universal format for data interchange between APIs and persistent local stores. The built-in 'json' module provides json.dumps() and json.loads() for converting Python dictionaries and lists to/from JSON strings, and json.dump() / json.load() for file streams.",
                            "concepts": ["json.loads() and json.dumps() string operations", "json.load() and json.dump() file stream operations", "Mapping JSON types to Python types (objects -> dicts, arrays -> lists)"],
                            "examples": [
                                "```python\nimport json\nlearner_profile = {\n    'name': 'Shreya',\n    'course': 'Python Masterclass',\n    'completed_modules': [1, 2, 3],\n    'gpa': 3.92\n}\n# Serialize to formatted JSON string\njson_str = json.dumps(learner_profile, indent=2)\n# Deserialize back to Python dict\nrestored = json.loads(json_str)\nprint('Course name:', restored['course'])\nprint('Completed count:', len(restored['completed_modules']))\n```\nExpected Output:\n```\nCourse name: Python Masterclass\nCompleted count: 3\n```"
                            ],
                            "exercises": ["Save user configuration settings to a config.json file", "Parse an API response payload and filter nested records"],
                            "takeaways": ["json.loads operates on strings; json.load operates on file streams", "JSON keys must be strings"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function extract_usernames(json_str) that parses a JSON list of user objects and returns a list of usernames.",
                                "starter_code": "import json\ndef extract_usernames(json_str):\n    pass\n",
                                "solution": "import json\ndef extract_usernames(json_str):\n    data = json.loads(json_str)\n    return [u['username'] for u in data if 'username' in u]\n",
                                "test_cases": [
                                    {"input": "'[{\"username\": \"alice\"}, {\"username\": \"bob\"}]'", "expected": "['alice', 'bob']"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Capstone Integration & Project Verification",
                            "explanation": "In this final lesson, integrate all the concepts mastered across the course: data structures, modular functions, object-oriented models, exception handling, and file operations. You will build and test the Personal Finance & Expense Tracker application, preparing it for submission and review.",
                            "concepts": ["End-to-end software architecture and separation of concerns", "Unit testing functions with assertions and test runners", "Packaging Python scripts with README documentation"],
                            "examples": [
                                "```python\n# Capstone Core Component Example\nclass ExpenseTracker:\n    def __init__(self):\n        self.expenses = []\n    def add(self, category: str, amount: float):\n        if amount <= 0:\n            raise ValueError('Amount must be positive')\n        self.expenses.append({'category': category, 'amount': amount})\n    def summary(self) -> dict:\n        res = {}\n        for e in self.expenses:\n            res[e['category']] = round(res.get(e['category'], 0) + e['amount'], 2)\n        return res\n\ntracker = ExpenseTracker()\ntracker.add('Food', 25.50)\ntracker.add('Books', 45.00)\ntracker.add('Food', 12.00)\nprint('Summary:', tracker.summary())\n```\nExpected Output:\n```\nSummary: {'Food': 37.5, 'Books': 45.0}\n```"
                            ],
                            "exercises": ["Implement export_to_json method for the ExpenseTracker class", "Write 3 unit tests verifying category calculations and error handling"],
                            "takeaways": ["Organize code into clear model, logic, and presentation layers", "Validate inputs before mutating application state"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Complete the tracker summary method to return {'total': float, 'count': int}.",
                                "starter_code": "def get_stats(transactions):\n    pass\n",
                                "solution": "def get_stats(transactions):\n    total = sum(t['amount'] for t in transactions)\n    return {'total': round(total, 2), 'count': len(transactions)}\n",
                                "test_cases": [
                                    {"input": "[{'amount': 10.5}, {'amount': 20.25}]", "expected": "{'total': 30.75, 'count': 2}"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "Why is 'with open(...) as f:' preferred over manually calling f.close()?",
                            "options": [
                                "It ensures the file is closed even if an unexpected exception occurs inside the block",
                                "It reads the file 10 times faster",
                                "It automatically converts text files into JSON objects",
                                "It encrypts file contents on disk"
                            ],
                            "correct_index": 0,
                            "explanation": "Context managers implement __enter__ and __exit__, guaranteeing file descriptor cleanup under all execution paths."
                        },
                        {
                            "prompt": "Which function converts a Python dictionary into a JSON formatted string?",
                            "options": [
                                "json.dumps()",
                                "json.loads()",
                                "json.parse()",
                                "json.stringify()"
                            ],
                            "correct_index": 0,
                            "explanation": "json.dumps() serializes a Python object into a JSON formatted string."
                        },
                        {
                            "prompt": "What file open mode appends new content to the end of an existing file without deleting it?",
                            "options": [
                                "'a' (append mode)",
                                "'w' (write mode)",
                                "'r' (read mode)",
                                "'x' (exclusive mode)"
                            ],
                            "correct_index": 0,
                            "explanation": "Mode 'a' appends data to the end of the file; mode 'w' truncates (erases) existing contents."
                        },
                        {
                            "prompt": "What is the primary indicator that software is ready for capstone completion?",
                            "options": [
                                "Core functional requirements pass test cases, handle errors, and adhere to clean architecture",
                                "The code file exceeds 5,000 lines",
                                "No functions are used in the codebase",
                                "The program requires administrator privileges to execute"
                            ],
                            "correct_index": 0,
                            "explanation": "Capstone completeness relies on robust functionality, test coverage, and modular design."
                        }
                    ]
                }
            ]
        }

    def _build_ml_course(self, level: str, duration: int, daily_min: int) -> Dict:
        return {
            "title": f"Machine Learning & AI Engineering ({level})",
            "description": f"A rigorous {duration}-day curriculum in applied machine learning: statistical data preprocessing, exploratory data analysis, supervised learning (regression, classification), model evaluation, unsupervised clustering, and an end-to-end predictive capstone.",
            "objectives": [
                "Prepare and clean messy real-world datasets using NumPy and Pandas",
                "Train supervised learning algorithms including Linear Regression, Decision Trees, and Random Forests",
                "Evaluate models rigorously using precision, recall, F1-score, ROC-AUC, and cross-validation",
                "Deploy a trained model pipeline to predict outcomes on unseen test datasets"
            ],
            "prerequisites": ["Python fundamentals (variables, functions, loops)", "Basic algebra and matrix intuition"],
            "capstone_project": {
                "title": "Customer Churn Prediction & Risk Modeling",
                "description": "Construct an end-to-end machine learning pipeline that ingests customer telecommunications telemetry, conducts exploratory data analysis, scales features, handles class imbalance, trains a Random Forest classifier, and evaluates performance using ROC-AUC and F1 scores.",
                "tasks": [
                    "Task 1: Load and clean raw CSV customer dataset, imputing missing values and encoding categories",
                    "Task 2: Perform feature engineering and scale numeric attributes using StandardScaler",
                    "Task 3: Train baseline Logistic Regression and tuned Random Forest models with 5-fold cross-validation",
                    "Task 4: Calculate confusion matrices, feature importances, and produce a business executive summary"
                ],
                "starter_code": "def evaluate_model(y_true, y_pred):\n    \"\"\"Calculate accuracy and precision.\"\"\"\n    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)\n    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)\n    prec = tp / (tp + fp) if (tp + fp) > 0 else 0\n    return {'precision': round(prec, 3)}\n",
                "deliverables": "Jupyter/Python script pipeline, trained weights, evaluation report, and feature importance chart."
            },
            "sources": [
                {"title": "Scikit-Learn Documentation", "url": "https://scikit-learn.org/stable/"},
                {"title": "Google Machine Learning Crash Course", "url": "https://developers.google.com/machine-learning/crash-course"}
            ],
            "modules": [
                {
                    "title": "Module 1: Mathematical Foundations & Data Preprocessing",
                    "description": "Understand vector mathematics, data frames, feature types, handling missing values, and categorical encoding.",
                    "duration_minutes": 180,
                    "objectives": ["Distinguish numeric and categorical features", "Handle missing data", "Implement one-hot and label encoding"],
                    "lessons": [
                        {
                            "title": "Lesson 1: The Machine Learning Workflow",
                            "explanation": "Machine learning algorithms learn patterns from empirical data rather than executing hard-coded rule-based logic. The workflow follows systematic stages: problem formulation, data collection, data cleaning, exploratory data analysis, train/test splitting, feature engineering, model training, hyperparameter tuning, evaluation, and deployment.",
                            "concepts": ["Features (X) vs Target labels (y)", "Supervised vs Unsupervised learning", "Train/Test split to avoid data leakage"],
                            "examples": [
                                "```python\n# Simulated train/test split\ndef train_test_split_simple(data, split_ratio=0.8):\n    cutoff = int(len(data) * split_ratio)\n    return data[:cutoff], data[cutoff:]\n\ndataset = list(range(100))\ntrain, test = train_test_split_simple(dataset)\nprint(f'Train count: {len(train)}, Test count: {len(test)}')\n```\nExpected Output:\n```\nTrain count: 80, Test count: 20\n```"
                            ],
                            "exercises": ["Separate independent features from target variables in a dict dataset", "Compute feature means and standard deviations"],
                            "takeaways": ["Never evaluate a model on the data it was trained on", "Data quality determines model accuracy bounds"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function split_features_target(records, target_key) returning (X, y).",
                                "starter_code": "def split_features_target(records, target_key):\n    pass\n",
                                "solution": "def split_features_target(records, target_key):\n    X = [{k: v for k, v in r.items() if k != target_key} for r in records]\n    y = [r[target_key] for r in records]\n    return X, y\n",
                                "test_cases": [
                                    {"input": "([{'a': 1, 'b': 2, 'target': 0}, {'a': 3, 'b': 4, 'target': 1}], 'target')", "expected": "([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], [0, 1])"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Handling Missing Values & Outliers",
                            "explanation": "Real-world datasets regularly suffer from missing values (NaN) and corrupted outliers. Imputation replaces missing entries using statistical metrics like mean, median, or mode. Outliers can distort regression algorithms and distance-based metrics.",
                            "concepts": ["Mean vs Median imputation robustness", "Z-score and Interquartile Range (IQR) outlier detection", "Data leakage prevention during imputation"],
                            "examples": [
                                "```python\n# Imputing missing values with median\nvalues = [10, 12, None, 14, 100, 15, None]\nclean = [v for v in values if v is not None]\nclean_sorted = sorted(clean)\nmedian = clean_sorted[len(clean_sorted) // 2]\nimputed = [v if v is not None else median for v in values]\nprint('Imputed dataset:', imputed)\n```\nExpected Output:\n```\nImputed dataset: [10, 12, 14, 14, 100, 15, 14]\n```"
                            ],
                            "exercises": ["Write an IQR outlier detector flagging entries beyond 1.5 * IQR", "Build a forward-fill imputer for sequential time data"],
                            "takeaways": ["Median is resistant to extreme outliers, whereas mean is skewed", "Calculate imputation statistics exclusively on training sets"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function impute_mean(nums) that replaces None values with the mean of non-None numbers.",
                                "starter_code": "def impute_mean(nums):\n    pass\n",
                                "solution": "def impute_mean(nums):\n    valid = [x for x in nums if x is not None]\n    avg = round(sum(valid) / len(valid), 2) if valid else 0\n    return [x if x is not None else avg for x in nums]\n",
                                "test_cases": [
                                    {"input": "[10, None, 20, None, 30]", "expected": "[10, 20.0, 20, 20.0, 30]"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Feature Scaling & Categorical Encoding",
                            "explanation": "Algorithms calculating distances (k-NN, SVM, Gradient Descent) are heavily biased if features have drastically different magnitudes (e.g. age: 20-70 vs salary: 30,000-150,000). Standardization scales variables to mean 0 and variance 1. Categorical variables are converted via One-Hot Encoding to avoid false ordinal hierarchy.",
                            "concepts": ["Standardization (Z-score) vs Min-Max Normalization", "One-Hot Encoding vs Ordinal Encoding", "The dummy variable trap in linear models"],
                            "examples": [
                                "```python\n# Min-Max normalization\nraw_data = [10, 20, 30, 40, 50]\nmin_v, max_v = min(raw_data), max(raw_data)\nscaled = [round((x - min_v) / (max_v - min_v), 2) for x in raw_data]\nprint('Min-Max scaled:', scaled)\n```\nExpected Output:\n```\nMin-Max scaled: [0.0, 0.25, 0.5, 0.75, 1.0]\n```"
                            ],
                            "exercises": ["Normalize a matrix of learner exam scores between 0 and 1", "Create one-hot encoded vectors for categories ['low', 'medium', 'high']"],
                            "takeaways": ["Scale test features using parameters learned strictly on training sets", "One-hot encoding creates binary indicators for nominal classes"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function min_max_scale(nums) returning floats scaled between 0.0 and 1.0.",
                                "starter_code": "def min_max_scale(nums):\n    pass\n",
                                "solution": "def min_max_scale(nums):\n    low, high = min(nums), max(nums)\n    if high == low:\n        return [0.0 for _ in nums]\n    return [round((x - low) / (high - low), 2) for x in nums]\n",
                                "test_cases": [
                                    {"input": "[0, 50, 100]", "expected": "[0.0, 0.5, 1.0]"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What is data leakage in machine learning preprocessing?",
                            "options": [
                                "Information from the test or validation set contaminates training feature calculations",
                                "Data is accidentally deleted during database migration",
                                "A model is stolen by an unauthorized API client",
                                "Floating point numbers lose precision in RAM"
                            ],
                            "correct_index": 0,
                            "explanation": "Data leakage occurs when test set signals leak into preprocessing, leading to over-optimistic validation scores that fail in production."
                        },
                        {
                            "prompt": "Why is One-Hot Encoding preferred over arbitrary integers for nominal categories (e.g. Red, Green, Blue)?",
                            "options": [
                                "It prevents algorithms from inferring a false mathematical ordering (e.g. Blue > Red)",
                                "It uses less disk space",
                                "It converts non-linear models into linear models",
                                "It guarantees 100% classification accuracy"
                            ],
                            "correct_index": 0,
                            "explanation": "Assigning 1, 2, 3 to colors introduces false geometric distances that distort linear weights and neural activations."
                        },
                        {
                            "prompt": "Which formula computes Min-Max normalization to the range [0, 1]?",
                            "options": [
                                "(x - min) / (max - min)",
                                "(x - mean) / standard_deviation",
                                "x / 100",
                                "log(x + 1)"
                            ],
                            "correct_index": 0,
                            "explanation": "Subtracting min and dividing by range (max - min) maps the minimum to 0 and maximum to 1."
                        },
                        {
                            "prompt": "Which summary statistic is least affected by extreme outlier data points?",
                            "options": [
                                "Median",
                                "Mean",
                                "Standard Deviation",
                                "Variance"
                            ],
                            "correct_index": 0,
                            "explanation": "The median represents the central 50th percentile rank, rendering it immune to solitary wild extreme values."
                        }
                    ]
                },
                {
                    "title": "Module 2: Supervised Learning — Linear & Logistic Regression",
                    "description": "Formulate predictive models using linear hypotheses, cost functions (MSE, Log Loss), and gradient descent optimization.",
                    "duration_minutes": 200,
                    "objectives": ["Implement Mean Squared Error and Binary Cross-Entropy", "Understand Gradient Descent optimization", "Interpret model weights and decision boundaries"],
                    "lessons": [
                        {
                            "title": "Lesson 1: Linear Regression & Ordinary Least Squares",
                            "explanation": "Linear regression models the relationship between continuous features and a continuous target through a linear equation y_hat = w1*x1 + ... + b. The Mean Squared Error (MSE) measures the average squared difference between predictions and actual values. OLS finds the weight vector minimizing MSE.",
                            "concepts": ["Hypothesis function y = WX + b", "Mean Squared Error (MSE) loss function", "Model assumptions (linearity, homoscedasticity, independence)"],
                            "examples": [
                                "```python\n# Calculating Mean Squared Error\ndef mse(y_true, y_pred):\n    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)\n\nactual = [100, 150, 200]\npredicted = [110, 140, 205]\nprint('MSE:', round(mse(actual, predicted), 2))\n```\nExpected Output:\n```\nMSE: 75.0\n```"
                            ],
                            "exercises": ["Fit a 1D trend line using covariance and variance", "Compute Root Mean Squared Error (RMSE)"],
                            "takeaways": ["MSE penalizes large outlier errors disproportionately due to squaring", "R-squared measures the proportion of variance explained by the model"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function compute_mse(actual, predicted) that returns the mean squared error rounded to 2 decimals.",
                                "starter_code": "def compute_mse(actual, predicted):\n    pass\n",
                                "solution": "def compute_mse(actual, predicted):\n    return round(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual), 2)\n",
                                "test_cases": [
                                    {"input": "([10, 20, 30], [12, 18, 33])", "expected": "5.67"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Logistic Regression & The Sigmoid Function",
                            "explanation": "Despite its name, Logistic Regression is a classification algorithm. It maps linear outputs (-inf to +inf) into probability estimates between 0 and 1 using the Sigmoid (logistic) function: sigma(z) = 1 / (1 + e^-z). Probabilities >= 0.5 are classified as the positive class (1). Log Loss penalizes confident wrong predictions severely.",
                            "concepts": ["The Sigmoid activation function", "Probability thresholding (default 0.5)", "Binary Cross-Entropy (Log Loss) cost function"],
                            "examples": [
                                "```python\nimport math\ndef sigmoid(z):\n    return 1 / (1 + math.exp(-z))\n\nprint('z = 0 -> prob:', sigmoid(0))\nprint('z = 2.5 -> prob:', round(sigmoid(2.5), 3))\nprint('z = -2.5 -> prob:', round(sigmoid(-2.5), 3))\n```\nExpected Output:\n```\nz = 0 -> prob: 0.5\nz = 2.5 -> prob: 0.924\nz = -2.5 -> prob: 0.076\n```"
                            ],
                            "exercises": ["Implement prediction threshold adjustment for high-risk scenarios", "Calculate log loss for a sample prediction vector"],
                            "takeaways": ["Sigmoid squashes any real value into the range (0, 1)", "Adjusting the decision threshold trades off precision against recall"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function predict_binary(z, threshold=0.5) that returns 1 if sigmoid(z) >= threshold else 0.",
                                "starter_code": "import math\ndef predict_binary(z, threshold=0.5):\n    pass\n",
                                "solution": "import math\ndef predict_binary(z, threshold=0.5):\n    prob = 1 / (1 + math.exp(-z))\n    return 1 if prob >= threshold else 0\n",
                                "test_cases": [
                                    {"input": "(1.5, 0.5)", "expected": "1"},
                                    {"input": "(-1.5, 0.5)", "expected": "0"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Gradient Descent Optimization",
                            "explanation": "Gradient Descent iteratively updates model parameters in the opposite direction of the gradient of the loss surface: w = w - learning_rate * dL/dw. Choosing an optimal learning rate is critical: too large causes divergence, while too small requires excessive training iterations.",
                            "concepts": ["Gradient as direction of steepest ascent", "The learning rate hyperparameter alpha", "Batch, Mini-batch, and Stochastic Gradient Descent (SGD)"],
                            "examples": [
                                "```python\n# 1D Gradient descent finding minimum of f(w) = w^2\nw = 10.0\nlearning_rate = 0.1\nfor step in range(5):\n    grad = 2 * w  # derivative of w^2\n    w = w - learning_rate * grad\n    print(f'Step {step+1}: w = {w:.3f}')\n```\nExpected Output:\n```\nStep 1: w = 8.000\nStep 2: w = 6.400\nStep 3: w = 5.120\nStep 4: w = 4.096\nStep 5: w = 3.277\n```"
                            ],
                            "exercises": ["Simulate gradient descent on a cost function with learning rate 0.01", "Visualize how learning rate 1.1 causes divergence"],
                            "takeaways": ["Gradient descent converges when the gradient approaches zero", "Feature scaling accelerates gradient descent convergence drastically"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function gradient_step(current_w, gradient, lr) returning the updated weight.",
                                "starter_code": "def gradient_step(current_w, gradient, lr):\n    pass\n",
                                "solution": "def gradient_step(current_w, gradient, lr):\n    return round(current_w - lr * gradient, 4)\n",
                                "test_cases": [
                                    {"input": "(5.0, 2.0, 0.1)", "expected": "4.8"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What range of values does the Sigmoid activation function output?",
                            "options": [
                                "Between 0 and 1 strictly",
                                "Between -1 and +1",
                                "Between 0 and infinity",
                                "Between -infinity and +infinity"
                            ],
                            "correct_index": 0,
                            "explanation": "Sigmoid maps the real line into (0, 1), making it ideal for modeling binary probabilities."
                        },
                        {
                            "prompt": "What happens during gradient descent if the learning rate is set too large?",
                            "options": [
                                "The optimization overshoots the minimum and may diverge",
                                "Training takes infinitely many epochs to start",
                                "The weights get stuck at zero permanently",
                                "The cost function is automatically smoothed"
                            ],
                            "correct_index": 0,
                            "explanation": "An oversized learning rate causes steps that overshoot the loss valley, causing oscillations or divergence."
                        },
                        {
                            "prompt": "Which metric evaluates continuous regression models rather than classification?",
                            "options": [
                                "Mean Squared Error (MSE)",
                                "Precision",
                                "F1 Score",
                                "Area Under ROC Curve (ROC-AUC)"
                            ],
                            "correct_index": 0,
                            "explanation": "MSE evaluates numerical prediction differences; the other three measure classification performance."
                        },
                        {
                            "prompt": "In linear regression, what does the bias term 'b' represent geometrically?",
                            "options": [
                                "The y-intercept when all features are zero",
                                "The slope of the steepest feature",
                                "The correlation coefficient",
                                "The standard error of residuals"
                            ],
                            "correct_index": 0,
                            "explanation": "In y = wx + b, b represents the value predicted when x=0 (the vertical intercept)."
                        }
                    ]
                },
                {
                    "title": "Module 3: Classification Evaluation Metrics",
                    "description": "Move beyond naive accuracy to evaluate classifiers using confusion matrices, precision, recall, F1-score, and ROC-AUC curves.",
                    "duration_minutes": 180,
                    "objectives": ["Construct and interpret confusion matrices", "Calculate precision, recall, and F1 score", "Assess model discrimination using ROC-AUC"],
                    "lessons": [
                        {
                            "title": "Lesson 1: The Accuracy Paradox & The Confusion Matrix",
                            "explanation": "When dealing with imbalanced datasets (e.g. credit fraud where 99.5% of transactions are legitimate), a model that blindly predicts 'Not Fraud' achieves 99.5% accuracy while providing zero real-world value. A confusion matrix breaks predictions into True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).",
                            "concepts": ["Accuracy paradox under severe class imbalance", "TP, FP, TN, and FN breakdown", "Type I (False Positive) vs Type II (False Negative) errors"],
                            "examples": [
                                "```python\n# Building a confusion matrix\ndef confusion_matrix(actual, pred):\n    tp = sum(1 for a, p in zip(actual, pred) if a == 1 and p == 1)\n    fp = sum(1 for a, p in zip(actual, pred) if a == 0 and p == 1)\n    tn = sum(1 for a, p in zip(actual, pred) if a == 0 and p == 0)\n    fn = sum(1 for a, p in zip(actual, pred) if a == 1 and p == 0)\n    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}\n\nact = [1, 1, 0, 0, 1]\nprd = [1, 0, 0, 1, 1]\nprint('Matrix:', confusion_matrix(act, prd))\n```\nExpected Output:\n```\nMatrix: {'TP': 2, 'FP': 1, 'TN': 1, 'FN': 1}\n```"
                            ],
                            "exercises": ["Classify medical screening errors as Type I or Type II", "Calculate overall accuracy from confusion matrix components"],
                            "takeaways": ["High accuracy on imbalanced data is frequently deceptive", "Type I and Type II errors carry different economic and safety costs"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function get_confusion(actual, pred) returning {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}.",
                                "starter_code": "def get_confusion(actual, pred):\n    pass\n",
                                "solution": "def get_confusion(actual, pred):\n    tp = sum(1 for a, p in zip(actual, pred) if a == 1 and p == 1)\n    fp = sum(1 for a, p in zip(actual, pred) if a == 0 and p == 1)\n    tn = sum(1 for a, p in zip(actual, pred) if a == 0 and p == 0)\n    fn = sum(1 for a, p in zip(actual, pred) if a == 1 and p == 0)\n    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}\n",
                                "test_cases": [
                                    {"input": "([1, 0, 1], [1, 1, 0])", "expected": "{'TP': 1, 'FP': 1, 'TN': 0, 'FN': 1}"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Precision, Recall & F1-Score",
                            "explanation": "Precision measures exactness: of all instances predicted positive, how many were truly positive? Precision = TP / (TP + FP). Recall (Sensitivity) measures completeness: of all actual positive instances, how many did the model catch? Recall = TP / (TP + FN). The F1-score is the harmonic mean of precision and recall.",
                            "concepts": ["Precision: TP / (TP + FP)", "Recall: TP / (TP + FN)", "F1-Score: 2 * (P * R) / (P + R)"],
                            "examples": [
                                "```python\ndef pr_f1(tp, fp, fn):\n    precision = tp / (tp + fp) if (tp + fp) else 0.0\n    recall = tp / (tp + fn) if (tp + fn) else 0.0\n    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0\n    return round(precision, 3), round(recall, 3), round(f1, 3)\n\np, r, f1 = pr_f1(tp=80, fp=20, fn=10)\nprint(f'Precision: {p}, Recall: {r}, F1: {f1}')\n```\nExpected Output:\n```\nPrecision: 0.8, Recall: 0.889, F1: 0.842\n```"
                            ],
                            "exercises": ["Determine whether fraud detection prioritizes precision or recall", "Calculate the F1 score for an email spam filter"],
                            "takeaways": ["Harmonic mean severely penalizes extreme divergence between precision and recall", "Tune thresholds based on business priorities"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function calc_f1(tp, fp, fn) returning the F1 score rounded to 2 decimal places.",
                                "starter_code": "def calc_f1(tp, fp, fn):\n    pass\n",
                                "solution": "def calc_f1(tp, fp, fn):\n    p = tp / (tp + fp) if (tp + fp) else 0\n    r = tp / (tp + fn) if (tp + fn) else 0\n    return round(2 * p * r / (p + r), 2) if (p + r) else 0.0\n",
                                "test_cases": [
                                    {"input": "(80, 20, 10)", "expected": "0.84"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: The ROC Curve & Area Under the Curve (AUC)",
                            "explanation": "A Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Recall) against False Positive Rate (FPR = FP / (FP + TN)) across all possible classification probability thresholds. The Area Under the Curve (ROC-AUC) measures overall class discrimination ability: 0.5 indicates random guessing, while 1.0 represents perfect separation.",
                            "concepts": ["True Positive Rate vs False Positive Rate", "ROC curve generation across varying thresholds", "Interpreting ROC-AUC score thresholds (0.5 to 1.0)"],
                            "examples": [
                                "```python\n# ROC-AUC intuitive interpretation\nprint('AUC 0.5: Random chance baseline (no discriminatory power)')\nprint('AUC 0.7-0.8: Acceptable discrimination')\nprint('AUC 0.8-0.9: Excellent discrimination')\nprint('AUC 0.9+: Outstanding discrimination')\n```\nExpected Output:\n```\nAUC 0.5: Random chance baseline (no discriminatory power)\nAUC 0.7-0.8: Acceptable discrimination\nAUC 0.8-0.9: Excellent discrimination\nAUC 0.9+: Outstanding discrimination\n```"
                            ],
                            "exercises": ["Plot TPR and FPR points for thresholds [0.2, 0.5, 0.8]", "Evaluate the effect of class imbalance on ROC curves"],
                            "takeaways": ["ROC-AUC is insensitive to class distribution changes", "PR curves are preferred when the minority positive class is extremely scarce"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function calculate_fpr(fp, tn) returning False Positive Rate rounded to 2 decimals.",
                                "starter_code": "def calculate_fpr(fp, tn):\n    pass\n",
                                "solution": "def calculate_fpr(fp, tn):\n    return round(fp / (fp + tn), 2) if (fp + tn) else 0.0\n",
                                "test_cases": [
                                    {"input": "(10, 90)", "expected": "0.1"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "In an oncology screening system where missing a cancer diagnosis is catastrophic, which metric must be maximized?",
                            "options": [
                                "Recall (Sensitivity)",
                                "Precision",
                                "Accuracy",
                                "Specificity"
                            ],
                            "correct_index": 0,
                            "explanation": "Recall minimizes False Negatives (missed diagnoses), ensuring almost all affected patients are flagged."
                        },
                        {
                            "prompt": "What does an ROC-AUC score of 0.5 represent?",
                            "options": [
                                "Performance equivalent to random guessing",
                                "Perfect classification separation",
                                "50% error rate on training data",
                                "Severe overfitting"
                            ],
                            "correct_index": 0,
                            "explanation": "An AUC of 0.5 corresponds to the diagonal line of chance, having zero discriminatory signal."
                        },
                        {
                            "prompt": "What is the formula for Precision?",
                            "options": [
                                "True Positives / (True Positives + False Positives)",
                                "True Positives / (True Positives + False Negatives)",
                                "True Positives / Total Predictions",
                                "(True Positives + True Negatives) / Total"
                            ],
                            "correct_index": 0,
                            "explanation": "Precision evaluates how many of the positively predicted instances were truly positive."
                        },
                        {
                            "prompt": "Why is the harmonic mean used for F1 score instead of an arithmetic mean?",
                            "options": [
                                "It heavily penalizes cases where one metric is near zero",
                                "It scales linearly with dataset size",
                                "It ignores false negatives entirely",
                                "It prevents division by zero mathematically"
                            ],
                            "correct_index": 0,
                            "explanation": "The harmonic mean pulls the average toward the lower value, preventing a model with 100% recall and 1% precision from scoring well."
                        }
                    ]
                },
                {
                    "title": "Module 4: Tree-Based Models & Ensemble Learning",
                    "description": "Understand non-linear decision boundaries with Decision Trees, Information Gain, Gini Impurity, Random Forests, and bagging.",
                    "duration_minutes": 200,
                    "objectives": ["Understand recursive binary splitting", "Calculate Gini Impurity and Information Gain", "Construct Random Forest ensembles"],
                    "lessons": [
                        {
                            "title": "Lesson 1: Decision Trees & Gini Impurity",
                            "explanation": "Decision Trees partition feature space into orthogonal rectangular regions through recursive binary splits. At each node, the algorithm evaluates all candidate features and split thresholds, selecting the split that maximizes purity (minimizes Gini Impurity or Entropy). Deep trees tend to overfit training data by memorizing noise.",
                            "concepts": ["Recursive binary splitting", "Gini Impurity formula: 1 - sum(p_i^2)", "Tree pruning and max_depth constraints"],
                            "examples": [
                                "```python\n# Calculating Gini Impurity for binary node\ndef gini(p1, p2):\n    return round(1 - (p1**2 + p2**2), 3)\n\nprint('Pure node (all class 1):', gini(1.0, 0.0))\nprint('50/50 split (maximum impurity):', gini(0.5, 0.5))\nprint('90/10 split:', gini(0.9, 0.1))\n```\nExpected Output:\n```\nPure node (all class 1): 0.0\n50/50 split (maximum impurity): 0.5\n90/10 split: 0.18\n```"
                            ],
                            "exercises": ["Calculate entropy for a multi-class node", "Determine the best split threshold for a 1D continuous feature"],
                            "takeaways": ["Gini impurity = 0 indicates a completely homogeneous (pure) node", "Limit max_depth and min_samples_split to prevent overfitting"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function calculate_gini(p1, p2) returning Gini impurity rounded to 2 decimals.",
                                "starter_code": "def calculate_gini(p1, p2):\n    pass\n",
                                "solution": "def calculate_gini(p1, p2):\n    return round(1.0 - (p1**2 + p2**2), 2)\n",
                                "test_cases": [
                                    {"input": "(0.5, 0.5)", "expected": "0.5"},
                                    {"input": "(1.0, 0.0)", "expected": "0.0"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Bagging & Random Forests",
                            "explanation": "Random Forest is an ensemble method that combines hundreds of decorrelated decision trees using Bootstrap Aggregating (Bagging). Each tree is trained on a random bootstrap sample of rows with replacement, and splits consider only a random subset of features (typically sqrt(p)). Predictions are aggregated via majority voting, reducing variance without increasing bias.",
                            "concepts": ["Bootstrap aggregating (sampling with replacement)", "Feature subsampling for tree decorrelation", "Majority voting and out-of-bag (OOB) error"],
                            "examples": [
                                "```python\n# Majority voting across 5 ensemble trees\ntree_predictions = [1, 1, 0, 1, 0]\nmajority = 1 if sum(tree_predictions) > len(tree_predictions) / 2 else 0\nprint('Ensemble prediction:', majority)\n```\nExpected Output:\n```\nEnsemble prediction: 1\n```"
                            ],
                            "exercises": ["Implement a bootstrap sampler taking N items with replacement", "Aggregate probability predictions across simulated classifiers"],
                            "takeaways": ["Ensembles reduce model variance significantly compared to solitary decision trees", "Random forests require minimal feature scaling"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function ensemble_vote(votes) returning the majority binary prediction (0 or 1).",
                                "starter_code": "def ensemble_vote(votes):\n    pass\n",
                                "solution": "def ensemble_vote(votes):\n    return 1 if sum(votes) >= len(votes) / 2 else 0\n",
                                "test_cases": [
                                    {"input": "[1, 1, 0, 0, 1]", "expected": "1"},
                                    {"input": "[0, 0, 1, 0]", "expected": "0"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Feature Importance Analysis",
                            "explanation": "Ensembles like Random Forests output feature importance scores by measuring the mean decrease in impurity caused by splits on each feature across all trees (MDI) or by measuring accuracy loss when feature columns are randomly permuted (Permutation Importance).",
                            "concepts": ["Mean Decrease in Impurity (MDI)", "Permutation Feature Importance", "Feature selection for production latency optimization"],
                            "examples": [
                                "```python\n# Sorting feature importances\nimportances = {'age': 0.12, 'tenure': 0.35, 'contract_type': 0.45, 'gender': 0.08}\nranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)\nfor feat, score in ranked:\n    print(f'{feat}: {score * 100:.1f}%')\n```\nExpected Output:\n```\ncontract_type: 45.0%\ntenure: 35.0%\nage: 12.0%\ngender: 8.0%\n```"
                            ],
                            "exercises": ["Drop features contributing under 2% to total importance", "Compare MDI against permutation importance on collinear data"],
                            "takeaways": ["Feature importance identifies high-impact drivers for business stakeholders", "Collinear features can split importance scores artificially"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function top_features(feat_dict, k) returning the top k feature names sorted by importance.",
                                "starter_code": "def top_features(feat_dict, k):\n    pass\n",
                                "solution": "def top_features(feat_dict, k):\n    return [k for k, v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True)[:k]]\n",
                                "test_cases": [
                                    {"input": "({'a': 0.1, 'b': 0.5, 'c': 0.4}, 2)", "expected": "['b', 'c']"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What key technique prevents individual trees in a Random Forest from being identical?",
                            "options": [
                                "Bootstrap row sampling and random feature subset selection at each split",
                                "Training each tree on a completely different dataset from different years",
                                "Using different programming languages for each tree",
                                "Inverting the labels on half of the trees"
                            ],
                            "correct_index": 0,
                            "explanation": "Bootstrap aggregating with random feature subspace selection ensures trees remain decorrelated."
                        },
                        {
                            "prompt": "What does a Gini Impurity of 0.0 at a tree node indicate?",
                            "options": [
                                "The node contains samples belonging exclusively to a single class (pure)",
                                "The node is split evenly 50/50 between two classes",
                                "The tree has reached an infinite depth loop",
                                "All predictions at the node are guaranteed to be wrong"
                            ],
                            "correct_index": 0,
                            "explanation": "Gini of 0 means 100% of samples in that partition belong to one class."
                        },
                        {
                            "prompt": "What is the primary risk of an unconstrained Decision Tree with no max_depth limit?",
                            "options": [
                                "Severe overfitting to training noise",
                                "Underfitting with high bias",
                                "Negative probability outputs",
                                "Division by zero during prediction"
                            ],
                            "correct_index": 0,
                            "explanation": "An unconstrained decision tree splits until every sample is isolated, creating extreme variance and memorizing noise."
                        },
                        {
                            "prompt": "How does a Random Forest aggregate individual tree outputs for a classification task?",
                            "options": [
                                "Majority voting across all individual tree predictions",
                                "Multiplying the tree outputs together",
                                "Selecting the prediction of the oldest tree",
                                "Taking the median tree index"
                            ],
                            "correct_index": 0,
                            "explanation": "For classification, Random Forests use majority voting (or mean predicted class probabilities) across all ensemble trees."
                        }
                    ]
                },
                {
                    "title": "Module 5: Unsupervised Learning & Dimensionality Reduction",
                    "description": "Discover latent clusters and patterns without labels using K-Means clustering, the Elbow method, and Principal Component Analysis (PCA).",
                    "duration_minutes": 180,
                    "objectives": ["Implement K-Means clustering", "Determine cluster count using inertia and the Elbow method", "Apply Principal Component Analysis for dimensionality reduction"],
                    "lessons": [
                        {
                            "title": "Lesson 1: K-Means Clustering Algorithm",
                            "explanation": "K-Means is an unsupervised clustering algorithm that partitions unlabeled data into K non-overlapping clusters. It alternates between two steps: assigning each point to its nearest centroid based on Euclidean distance, and recalculating centroids as the geometric mean of assigned points, until convergence.",
                            "concepts": ["Centroid initialization and convergence", "Euclidean distance calculation", "Within-Cluster Sum of Squares (Inertia)"],
                            "examples": [
                                "```python\n# 2D Euclidean distance\nimport math\ndef euclidean(p1, p2):\n    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))\n\nprint('Distance (0,0) to (3,4):', euclidean((0,0), (3,4)))\n```\nExpected Output:\n```\nDistance (0,0) to (3,4): 5.0\n```"
                            ],
                            "exercises": ["Assign points to 2 initial centroids and recompute centers", "Implement K-Means++ initialization logic"],
                            "takeaways": ["K-Means assumes spherical clusters of comparable size", "Feature scaling is mandatory because K-Means relies on distance metrics"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function assign_cluster(point, centroids) returning the index of the closest centroid.",
                                "starter_code": "import math\ndef assign_cluster(point, centroids):\n    pass\n",
                                "solution": "import math\ndef assign_cluster(point, centroids):\n    dists = [sum((a - b)**2 for a, b in zip(point, c)) for c in centroids]\n    return dists.index(min(dists))\n",
                                "test_cases": [
                                    {"input": "([1, 1], [[0, 0], [10, 10]])", "expected": "0"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: Determining K with The Elbow Method",
                            "explanation": "Selecting the optimal number of clusters K is challenging because inertia decreases monotonically as K increases. The Elbow Method plots inertia against K: the 'elbow' point marks diminishing returns where adding additional clusters yields marginal inertia reduction.",
                            "concepts": ["Inertia / WCSS tracking across candidate K values", "Identifying the inflection 'elbow' point", "Silhouette analysis for cluster cohesion and separation"],
                            "examples": [
                                "```python\n# Detecting inertia drop\ninertias = {1: 1200, 2: 450, 3: 210, 4: 180, 5: 160}\nfor k in range(2, 6):\n    drop = inertias[k-1] - inertias[k]\n    print(f'K={k-1} -> K={k}: Inertia dropped by {drop}')\n```\nExpected Output:\n```\nK=1 -> K=2: Inertia dropped by 750\nK=2 -> K=3: Inertia dropped by 240\nK=3 -> K=4: Inertia dropped by 30\nK=4 -> K=5: Inertia dropped by 20\n```"
                            ],
                            "exercises": ["Compute percentage variance explained across candidate K values", "Calculate silhouette coefficients for clustered data points"],
                            "takeaways": ["The elbow identifies the sweet spot between compactness and complexity", "Silhouette scores range from -1 (misclustered) to +1 (dense, distinct clusters)"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function find_largest_drop(inertias_list) returning the index of the transition with the greatest drop.",
                                "starter_code": "def find_largest_drop(inertias_list):\n    pass\n",
                                "solution": "def find_largest_drop(inertias_list):\n    drops = [inertias_list[i] - inertias_list[i+1] for i in range(len(inertias_list)-1)]\n    return drops.index(max(drops)) + 1\n",
                                "test_cases": [
                                    {"input": "[1000, 400, 200, 150]", "expected": "1"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Dimensionality Reduction via PCA",
                            "explanation": "High-dimensional datasets suffer from the curse of dimensionality, where data becomes sparse and computation expensive. Principal Component Analysis (PCA) performs an orthogonal linear transformation to project data onto principal component axes that capture maximum variance, enabling 2D/3D visualization and noise reduction.",
                            "concepts": ["Eigenvalues, eigenvectors, and variance capture", "Curse of dimensionality in high dimensions", "Explained variance ratio and scree plots"],
                            "examples": [
                                "```python\n# Cumulative explained variance interpretation\nvar_ratio = [0.65, 0.22, 0.08, 0.05]\ncumulative = []\ntotal = 0\nfor v in var_ratio:\n    total += v\n    cumulative.append(round(total, 2))\nprint('Cumulative variance:', cumulative)\n```\nExpected Output:\n```\nCumulative variance: [0.65, 0.87, 0.95, 1.0]\n```"
                            ],
                            "exercises": ["Project 3D points onto their top 2 principal components", "Determine the minimum components required to retain 95% variance"],
                            "takeaways": ["Data must be standardized (mean 0, variance 1) prior to running PCA", "Top principal components capture the majority of dataset information"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function components_for_variance(ratios, threshold=0.9) returning the min component count needed to reach threshold.",
                                "starter_code": "def components_for_variance(ratios, threshold=0.9):\n    pass\n",
                                "solution": "def components_for_variance(ratios, threshold=0.9):\n    total = 0\n    for i, r in enumerate(ratios, 1):\n        total += r\n        if total >= threshold:\n            return i\n    return len(ratios)\n",
                                "test_cases": [
                                    {"input": "([0.5, 0.3, 0.15, 0.05], 0.8)", "expected": "2"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What distinguishes unsupervised learning from supervised learning?",
                            "options": [
                                "Unsupervised learning discovers structure in data without explicit ground-truth target labels",
                                "Unsupervised learning requires no data inputs",
                                "Unsupervised learning guarantees 100% accuracy",
                                "Unsupervised learning runs only on quantum hardware"
                            ],
                            "correct_index": 0,
                            "explanation": "Unsupervised learning discovers intrinsic clusters, distributions, and representations without target label supervision."
                        },
                        {
                            "prompt": "Why is feature scaling essential before running K-Means clustering?",
                            "options": [
                                "K-Means calculates Euclidean distances; unscaled features with large ranges dominate distance measurements",
                                "K-Means only accepts inputs between 0 and 1",
                                "Unscaled data causes infinite loops in Python",
                                "Scaling removes missing values automatically"
                            ],
                            "correct_index": 0,
                            "explanation": "Because distance calculations treat all axis units equally, an unscaled feature like 'annual income' dwarfs 'years of schooling'."
                        },
                        {
                            "prompt": "What does Principal Component Analysis (PCA) preserve during projection?",
                            "options": [
                                "Maximum possible variance in the orthogonal coordinate axes",
                                "The exact original feature column names",
                                "Categorical labels",
                                "The total number of feature columns"
                            ],
                            "correct_index": 0,
                            "explanation": "PCA selects orthogonal projection axes (eigenvectors) that maximize the variance of projected points."
                        },
                        {
                            "prompt": "In the Elbow Method for K-Means, what does the 'elbow' represent on the plot?",
                            "options": [
                                "The point where the rate of decrease in within-cluster inertia sharply slows down",
                                "The point where inertia drops to exactly zero",
                                "The point where training time becomes infinite",
                                "The point where accuracy becomes 100%"
                            ],
                            "correct_index": 0,
                            "explanation": "The elbow indicates where adding more clusters yields diminishing returns in cluster compactness."
                        }
                    ]
                },
                {
                    "title": "Module 6: Capstone Pipeline & Model Deployment",
                    "description": "Construct an end-to-end Machine Learning pipeline integrating preprocessing, cross-validation, hyperparameter tuning, model persistence, and deployment verification.",
                    "duration_minutes": 240,
                    "objectives": ["Build a scikit-learn Pipeline architecture", "Execute 5-fold cross-validation without leakage", "Deliver the Customer Churn capstone project"],
                    "lessons": [
                        {
                            "title": "Lesson 1: Scikit-Learn Pipelines & Transformers",
                            "explanation": "Pipelines chain data preprocessing and estimators into a unified, atomic object. Applying .fit() on a Pipeline fits transformers strictly on training data and transforms test data during .predict(), eliminating data leakage and ensuring reproducible inference in production.",
                            "concepts": ["Pipeline and ColumnTransformer architecture", "Fit vs Transform distinctions", "Eliminating train/test leakage with automated pipelines"],
                            "examples": [
                                "```python\n# Conceptual Pipeline workflow\nclass MockPipeline:\n    def __init__(self, steps):\n        self.steps = steps\n    def fit(self, X, y):\n        curr = X\n        for name, step in self.steps[:-1]:\n            curr = step.fit_transform(curr)\n        self.steps[-1][1].fit(curr, y)\n    def predict(self, X):\n        curr = X\n        for name, step in self.steps[:-1]:\n            curr = step.transform(curr)\n        return self.steps[-1][1].predict(curr)\nprint('Pipeline architecture established')\n```\nExpected Output:\n```\nPipeline architecture established\n```"
                            ],
                            "exercises": ["Chain an imputer, standard scaler, and logistic regression into a pipeline", "Validate inference latency per sample"],
                            "takeaways": ["Pipelines ensure identical transformations are applied at train and test time", "Simplifies model serialization and cloud deployment"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function transform_chain(data, funcs) that applies a list of transformation functions sequentially to data.",
                                "starter_code": "def transform_chain(data, funcs):\n    pass\n",
                                "solution": "def transform_chain(data, funcs):\n    curr = data\n    for f in funcs:\n        curr = f(curr)\n    return curr\n",
                                "test_cases": [
                                    {"input": "(5, [lambda x: x + 1, lambda x: x * 2])", "expected": "12"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 2: K-Fold Cross-Validation & Hyperparameter Tuning",
                            "explanation": "A single train/test split can yield misleading metrics if the split happens to be lucky or unlucky. K-Fold Cross-Validation divides the dataset into K equal folds, training on K-1 folds and validating on the remaining fold, repeating K times to report mean score and variance. Grid Search and Random Search explore hyperparameter combinations systematically.",
                            "concepts": ["Stratified K-Fold for class imbalance", "GridSearchCV vs RandomizedSearchCV", "Bias-variance tradeoff tuning"],
                            "examples": [
                                "```python\n# Simulated cross validation scoring\nfold_scores = [0.82, 0.85, 0.83, 0.86, 0.84]\nmean_score = sum(fold_scores) / len(fold_scores)\nvariance = sum((s - mean_score)**2 for s in fold_scores) / len(fold_scores)\nprint(f'CV Mean Accuracy: {mean_score:.3f} (+/- {variance**0.5:.3f})')\n```\nExpected Output:\n```\nCV Mean Accuracy: 0.840 (+/- 0.014)\n```"
                            ],
                            "exercises": ["Implement stratified fold indexing ensuring balanced class ratios", "Tune max_depth across [3, 5, 10, 20]"],
                            "takeaways": ["Report both the mean cross-validation score and standard deviation", "Stratified folds preserve target class ratios in each split"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function mean_cv_score(scores) returning the average score rounded to 3 decimal places.",
                                "starter_code": "def mean_cv_score(scores):\n    pass\n",
                                "solution": "def mean_cv_score(scores):\n    return round(sum(scores) / len(scores), 3) if scores else 0.0\n",
                                "test_cases": [
                                    {"input": "[0.81, 0.83, 0.85]", "expected": "0.83"}
                                ]
                            }
                        },
                        {
                            "title": "Lesson 3: Capstone Integration — Customer Churn Classifier",
                            "explanation": "Bring together the complete predictive modeling lifecycle: ingest telemetry, clean missing indicators, scale numeric features, train an ensemble model, tune hyperparameters with cross-validation, and evaluate using ROC-AUC and F1-score to deliver the final Capstone.",
                            "concepts": ["End-to-end integration checklist", "Exporting model artifacts with joblib/pickle", "Generating executive business insights from feature importances"],
                            "examples": [
                                "```python\n# Capstone metric report\nmetrics = {'ROC-AUC': 0.892, 'F1-Score': 0.814, 'Precision': 0.845, 'Recall': 0.785}\nfor m, v in metrics.items():\n    print(f'{m}: {v:.3f}')\n```\nExpected Output:\n```\nROC-AUC: 0.892\nF1-Score: 0.814\nPrecision: 0.845\nRecall: 0.785\n```"
                            ],
                            "exercises": ["Generate an end-to-end evaluation report on the capstone dataset", "Identify top 3 drivers of customer churn from feature weights"],
                            "takeaways": ["Connect machine learning metrics to concrete business impact", "Verify reproducibility by seeding random state variables"],
                            "coding_exercise": {
                                "language": "python",
                                "instructions": "Write a function is_capstone_ready(auc, f1, threshold_auc=0.8, threshold_f1=0.75) returning True if both thresholds are met.",
                                "starter_code": "def is_capstone_ready(auc, f1, threshold_auc=0.8, threshold_f1=0.75):\n    pass\n",
                                "solution": "def is_capstone_ready(auc, f1, threshold_auc=0.8, threshold_f1=0.75):\n    return auc >= threshold_auc and f1 >= threshold_f1\n",
                                "test_cases": [
                                    {"input": "(0.85, 0.78)", "expected": "True"},
                                    {"input": "(0.79, 0.80)", "expected": "False"}
                                ]
                            }
                        }
                    ],
                    "questions": [
                        {
                            "prompt": "What is the primary benefit of wrapping data preprocessing and estimators inside a Pipeline?",
                            "options": [
                                "It guarantees preprocessing transforms are learned solely from training data and applied consistently during prediction",
                                "It forces code to run on GPU hardware",
                                "It compresses model files to under 1 KB",
                                "It eliminates the need for test data"
                            ],
                            "correct_index": 0,
                            "explanation": "Pipelines prevent data leakage by strictly fitting transformers on training partitions during cross-validation."
                        },
                        {
                            "prompt": "What does Stratified K-Fold cross-validation guarantee that standard K-Fold does not?",
                            "options": [
                                "Each fold preserves approximately the same percentage of target class samples as the complete dataset",
                                "Each fold runs in parallel threads",
                                "The model weights are never updated",
                                "The features are standardized automatically"
                            ],
                            "correct_index": 0,
                            "explanation": "Stratification ensures minority classes are evenly distributed across all validation folds."
                        },
                        {
                            "prompt": "Why should random_state (seed) parameters be set when splitting datasets and training models?",
                            "options": [
                                "To guarantee reproducible results across separate experimental runs",
                                "To encrypt the model parameters",
                                "To speed up gradient descent",
                                "To avoid memory allocation errors"
                            ],
                            "correct_index": 0,
                            "explanation": "Fixing the random seed ensures stochastic operations produce identical splits and initializations."
                        },
                        {
                            "prompt": "Which serialized format is standard for persisting Scikit-Learn pipelines to disk?",
                            "options": [
                                "Joblib or Pickle (.joblib, .pkl)",
                                "Raw HTML (.html)",
                                "Plain CSV (.csv)",
                                "Markdown (.md)"
                            ],
                            "correct_index": 0,
                            "explanation": "joblib and pickle serialize Python object graphs containing learned weights and pipeline steps."
                        }
                    ]
                }
            ]
        }

    def _build_sql_course(self, level: str, duration: int, daily_min: int) -> Dict:
        return self._build_generic_course("SQL & Database Engineering", level, duration, daily_min)

    def _build_web_course(self, level: str, duration: int, daily_min: int) -> Dict:
        return self._build_generic_course("Full-Stack Web Development", level, duration, daily_min)

    def _build_java_course(self, level: str, duration: int, daily_min: int) -> Dict:
        return self._build_generic_course("Java Programming & OOP", level, duration, daily_min)

    def _build_dsa_course(self, level: str, duration: int, daily_min: int) -> Dict:
        return self._build_generic_course("Data Structures & Algorithms", level, duration, daily_min)

    def _build_generic_course(self, topic: str, level: str, duration: int, daily_min: int) -> Dict:
        """Create a complete 6-module course for any custom topic."""
        modules = []
        themes = [
            ("Core Foundations & Principles", f"Establish core vocabulary, fundamental principles, environment setup, and prerequisites for {topic}."),
            ("Essential Methods & Workflows", f"Master the primary mechanisms, patterns, and day-to-day practices used in {topic}."),
            ("Practical Applications & Case Studies", f"Work through realistic scenarios, concrete examples, and common architectural patterns in {topic}."),
            ("Advanced Techniques & Optimization", f"Analyze edge cases, performance considerations, efficiency, and debugging strategies in {topic}."),
            ("Testing, Security & Best Practices", f"Apply verification techniques, automated validation, error handling, and security standards for {topic}."),
            ("Capstone Project & Portfolio Synthesis", f"Integrate all concepts into a complete, documented capstone project demonstrating mastery in {topic}.")
        ]
        
        for i, (mtitle, desc) in enumerate(themes, 1):
            lessons = []
            for j in range(1, 4):
                ltitle = f"Lesson {i}.{j}: {mtitle} Part {j}"
                explanation = f"This comprehensive lesson provides in-depth exploration of {mtitle.lower()} within {topic}. Learners will study core mechanisms, terminology, and operational principles. We compare optimal practices against common anti-patterns, dissecting the conceptual architecture step by step."
                lessons.append({
                    "title": ltitle,
                    "explanation": explanation,
                    "concepts": [
                        f"Fundamental concept {j} in {mtitle}",
                        f"Operational trade-offs and best practices in {topic}"
                    ],
                    "examples": [
                        f"Worked example demonstrating {mtitle} applied to a realistic {topic} scenario with expected outputs."
                    ],
                    "exercises": [
                        f"Guided exercise applying the principles of {mtitle} to sample data.",
                        f"Independent challenge building a complete solution for {topic}."
                    ],
                    "takeaways": [
                        f"Key insight regarding {mtitle} and its importance to {topic}.",
                        "Always verify assumptions with test cases and official documentation."
                    ],
                    "coding_exercise": {
                        "language": "python",
                        "instructions": f"Implement a core helper function for {topic}.",
                        "starter_code": "def process_data(input_data):\n    # Return transformed result\n    pass\n",
                        "solution": "def process_data(input_data):\n    return [x for x in input_data if x]\n",
                        "test_cases": [
                            {"input": "[1, 0, 2]", "expected": "[1, 2]"}
                        ]
                    }
                })
            
            questions = [
                {
                    "prompt": f"What is the recommended first step when applying {mtitle.lower()} in {topic}?",
                    "options": [
                        "Define the objective, evaluate inputs, and verify underlying assumptions",
                        "Skip verification and deploy directly to production",
                        "Memorize one specific script without understanding the concepts",
                        "Ignore error handling and security guidelines"
                    ],
                    "correct_index": 0,
                    "explanation": "A disciplined engineering workflow starts with clear requirements and assumption validation."
                },
                {
                    "prompt": f"Why is systematic testing essential when working with {mtitle.lower()}?",
                    "options": [
                        "It detects edge cases, prevents regressions, and validates correctness",
                        "It replaces the need for understanding the core concepts",
                        "It slows down software execution permanently",
                        "It only benefits theoretical research projects"
                    ],
                    "correct_index": 0,
                    "explanation": "Systematic testing ensures reliable software execution across unexpected inputs and edge cases."
                },
                {
                    "prompt": f"How should discrepancies between expected and actual results in {mtitle.lower()} be investigated?",
                    "options": [
                        "Isolate variables systematically and test each assumption independently",
                        "Change multiple parameters simultaneously at random",
                        "Assume the testing framework is defective",
                        "Delete the test case from the test suite"
                    ],
                    "correct_index": 0,
                    "explanation": "Scientific debugging isolates one variable at a time to identify the exact root cause."
                },
                {
                    "prompt": f"Which documentation source is most authoritative when researching {topic}?",
                    "options": [
                        "Official specifications, vendor documentation, and peer-reviewed guides",
                        "Anonymous unverified social media posts",
                        "Deprecations from legacy versions with no updates",
                        "AI hallucinated sample snippets with no verification"
                    ],
                    "correct_index": 0,
                    "explanation": "Official documentation and peer-reviewed specifications provide verified authoritative standards."
                }
            ]
            
            modules.append({
                "title": f"Module {i}: {mtitle}",
                "description": desc,
                "duration_minutes": max(60, round(duration * daily_min / 6)),
                "objectives": [f"Master {mtitle.lower()}", f"Apply best practices in {topic}"],
                "lessons": lessons,
                "questions": questions
            })

        return {
            "title": f"{topic}: Complete {level} Specialization",
            "description": f"A comprehensive {duration}-day curriculum for {topic} ({level} level) featuring 6 structured modules, interactive coding exercises, video tutorials, and a practical capstone project.",
            "objectives": [
                f"Build a rigorous foundation in {topic}",
                "Apply core concepts through guided and independent hands-on exercises",
                "Evaluate performance, troubleshoot errors, and adhere to best practices",
                "Complete a comprehensive capstone project suitable for professional portfolios"
            ],
            "prerequisites": ["General computer literacy and motivation to learn"],
            "capstone_project": {
                "title": f"{topic} Comprehensive Capstone Project",
                "description": f"Design, build, and document an end-to-end practical application implementing the concepts learned throughout this {topic} course.",
                "tasks": [
                    "Task 1: Requirements specification and architecture design",
                    "Task 2: Core implementation with error handling",
                    "Task 3: Automated verification and testing",
                    "Task 4: Documentation and deployment packaging"
                ],
                "starter_code": "# Capstone Starter Template\ndef run_project():\n    print('Capstone project initialized')\n",
                "deliverables": "Working application source code, verification test suite, and project summary documentation."
            },
            "sources": [],
            "modules": modules
        }
