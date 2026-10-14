"""Gemini-backed course creation and course-aware tutoring.

Live mode uses Google's Gemini API with Google Search grounding when available.
Without a key, the app clearly reports that it is using local sample material.
"""
import json
import re


class AIService:
    def __init__(self, api_key='', model='gemini-2.5-flash', demo=True):
        self.api_key = (api_key or '').strip()
        self.model = model or 'gemini-2.5-flash'
        self.demo = bool(demo) or not self.api_key
        self.client = None
        self.types = None
        if self.api_key and not self.demo:
            try:
                from google import genai
                from google.genai import types
                self.types = types
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def _generate_text(self, prompt, use_google_search=True):
        if not self.client:
            return None
        try:
            config = {'temperature': 0.2}
            if use_google_search and self.types:
                config['tools'] = [self.types.Tool(google_search=self.types.GoogleSearch())]
            response = self.client.models.generate_content(
                model=self.model, contents=prompt,
                config=self.types.GenerateContentConfig(**config) if self.types else config,
            )
            return (response.text or '').strip() or None
        except Exception:
            # Some model/API combinations do not support Search grounding; retry with Gemini only.
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=self.types.GenerateContentConfig(temperature=0.2) if self.types else {'temperature': 0.2},
                )
                return (response.text or '').strip() or None
            except Exception:
                return None

    @staticmethod
    def _parse_json(raw):
        if not raw:
            return None
        raw = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw.strip(), flags=re.I)
        try:
            return json.loads(raw)
        except Exception:
            start, end = raw.find('{'), raw.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except Exception:
                    return None
        return None

    def reply(self, state, user_message, history):
        if self.client:
            prompt = f'''You are IntelliCourse's course-planning assistant. Extract preferences only when the learner states them clearly. Keep prior valid values.
Return ONLY JSON: {{"state":{{"topic":"","learning_goal":"","skill_level":"Beginner|Intermediate|Advanced","duration_days":30,"daily_minutes":60,"learning_style":"Theory|Practical|Mixed","desired_outcome":"","language":"English"}},"reply":"short helpful reply","ready":false}}.
Required for ready: topic, skill_level, duration_days, daily_minutes, learning_style. Ask exactly one concise question about the next missing field. Parse durations such as 6 weeks as 42 days and 1 hour as 60 minutes. Do not treat a question as a preference update. If ready, summarize the choices and ask the learner to generate or customize.
Current profile: {json.dumps(state, ensure_ascii=False)}
Recent conversation: {json.dumps(history[-10:], ensure_ascii=False)}
New learner message: {user_message}'''
            data = self._parse_json(self._generate_text(prompt, use_google_search=False))
            if isinstance(data, dict) and isinstance(data.get('state'), dict) and isinstance(data.get('reply'), str):
                data['state'] = self._clean_state({**(state or {}), **data['state']})
                data['ready'] = self._is_ready(data['state'])
                return data
        return self._demo_reply(state, user_message)

    @staticmethod
    def _clean_state(state):
        s = dict(state or {})
        for key in ('topic', 'learning_goal', 'desired_outcome', 'language', 'learning_style', 'skill_level'):
            if isinstance(s.get(key), str): s[key] = s[key].strip()[:500]
        for key, default, maximum in [('duration_days', 30, 365), ('daily_minutes', 60, 1440)]:
            try: s[key] = max(1, min(maximum, int(s.get(key, default)))) if s.get(key) else None
            except (ValueError, TypeError): s[key] = None
        if s.get('skill_level'):
            level = s['skill_level'].lower()
            s['skill_level'] = next((x for x in ('Beginner','Intermediate','Advanced') if x.lower() in level), s['skill_level'].title())
        if s.get('learning_style'):
            low = s['learning_style'].lower()
            s['learning_style'] = 'Mixed' if 'mix' in low else ('Practical' if 'pract' in low else ('Theory' if 'theory' in low else s['learning_style'].title()))
        return s

    def generate_course(self, prefs):
        prefs = self._clean_state(prefs)
        if self.client:
            prompt = f'''Create a COMPLETE, accurate, useful course for the learner below. Use Google Search grounding to verify current facts, official documentation, and recommended learning order. Prefer official documentation (MDN, Python docs, Java docs, Google for Developers, scikit-learn docs, etc.) and do not invent URLs, statistics, APIs, or citations. If web sources disagree, state the established consensus. Do not copy text; explain in original beginner-friendly language.

Learner profile: {json.dumps(prefs, ensure_ascii=False)}

Return ONLY one valid JSON object with this exact structure:
{{"title":"specific title","description":"specific overview","objectives":["..."],"sources":[{{"title":"source name","url":"https://..."}}],"modules":[{{"title":"specific module title","description":"what and why","duration_minutes":120,"objectives":["..."],"lessons":[{{"title":"specific lesson title","explanation":"teach the concept accurately in 2-5 paragraphs, define terms, explain why/how, include caveats","concepts":["specific concept and explanation"],"examples":["complete concrete example with expected result where relevant"],"exercises":["actionable exercise with clear deliverable"],"takeaways":["specific fact or skill"]}}],"questions":[{{"prompt":"unambiguous question","options":["A","B","C","D"],"correct_index":0,"explanation":"why the answer is correct and other tempting choices are wrong"}}]}}]}}

Requirements: exactly 6 modules; 3 or 4 lessons per module; 4 multiple-choice questions per module; all 4 options must be distinct; correct_index must be 0,1,2,or 3; at least 2 detailed examples and 2 meaningful exercises per lesson; no repeated generic lesson text; module order must build skills progressively; use the requested language and level; fit the total plan to {prefs.get('duration_days',30)} days and {prefs.get('daily_minutes',60)} minutes/day. Include a practical capstone in the final module. For programming topics, examples must contain valid code in the requested language and explain expected output. For non-programming topics, include concrete worked examples. Keep the course tightly specific to the topic, not a generic study-skills course. Include 3-6 real reliable sources with URLs. JSON only, no Markdown fences.'''
            data = self._parse_json(self._generate_text(prompt, use_google_search=True))
            if self.validate_course(data):
                data['sources'] = self._valid_sources(data.get('sources', []))
                return data, 'gemini_google_search' if data['sources'] else 'gemini'
        return self._sample_course(prefs), 'demo'

    @staticmethod
    def _valid_sources(sources):
        result = []
        for item in sources if isinstance(sources, list) else []:
            if isinstance(item, dict) and isinstance(item.get('title'), str) and isinstance(item.get('url'), str) and item['url'].startswith('https://'):
                result.append({'title': item['title'][:200], 'url': item['url'][:500]})
        return result[:8]

    @staticmethod
    def validate_course(data):
        if not isinstance(data, dict) or not isinstance(data.get('title'), str) or not data['title'].strip(): return False
        modules = data.get('modules')
        if not isinstance(modules, list) or len(modules) < 4: return False
        if not isinstance(data.get('objectives'), list) or len(data['objectives']) < 2: return False
        lesson_count = 0
        for module in modules:
            if not isinstance(module, dict) or not isinstance(module.get('title'), str) or not module['title'].strip(): return False
            lessons = module.get('lessons')
            if not isinstance(lessons, list) or len(lessons) < 2: return False
            lesson_count += len(lessons)
            for lesson in lessons:
                if not isinstance(lesson, dict) or not lesson.get('title') or not isinstance(lesson.get('explanation'), str) or len(lesson['explanation'].strip()) < 50: return False
                for field in ('concepts','examples','exercises','takeaways'):
                    if not isinstance(lesson.get(field), list) or not lesson[field]: return False
            questions = module.get('questions')
            if not isinstance(questions, list) or len(questions) < 3: return False
            for q in questions:
                if not isinstance(q, dict) or not isinstance(q.get('prompt'), str) or not q['prompt'].strip(): return False
                opts = q.get('options')
                if not isinstance(opts, list) or len(opts) != 4 or len({str(x).strip() for x in opts}) != 4: return False
                if not isinstance(q.get('correct_index'), int) or q['correct_index'] not in range(4): return False
                if not isinstance(q.get('explanation'), str) or not q['explanation'].strip(): return False
        return lesson_count >= 8

    def tutor(self, question, lesson=None, mode='answer'):
        context = json.dumps(lesson or {}, ensure_ascii=False)
        if self.client:
            raw = self._generate_text(f'''You are an accurate, patient subject tutor inside an online course. The user is asking about their actual course. Use the provided course and lesson context first. Answer the exact question directly, teach step by step if asked, and give a concrete example where useful. If the context is insufficient, use Google Search grounding and prefer official documentation. Never pretend an unsupported statement is certain; distinguish facts from assumptions. If the question is unrelated to the course, still answer helpfully. Do not merely repeat the question. Mode: {mode}. Course/lesson context JSON: {context}. Learner question: {question}
Return a concise but complete answer in readable Markdown. For code, use fenced code blocks.''', use_google_search=True)
            if raw: return raw, 'gemini_google_search'
        return self._demo_tutor(question, lesson, mode), 'demo'

    def _demo_reply(self, state, message):
        s = self._clean_state(state); m = message.strip(); low = m.lower()
        if not s.get('topic'):
            s['topic'] = m[:180]; s['learning_goal'] = m
            reply = f"Great, we'll focus on **{s['topic']}**. What is your current level: beginner, intermediate, or advanced?"
        elif not s.get('skill_level'):
            s['skill_level'] = next((x.title() for x in ['beginner','intermediate','advanced'] if x in low), 'Beginner')
            reply = 'How long do you want to study? For example, 14 days, 30 days, or 8 weeks.'
        elif not s.get('duration_days'):
            nums = re.search(r'\d+', m); s['duration_days'] = int(nums.group()) * (7 if 'week' in low and nums else 1) if nums else 30
            reply = 'How much time can you study each day? For example, 30 minutes or 1 hour.'
        elif not s.get('daily_minutes'):
            nums = re.search(r'\d+', m)
            s['daily_minutes'] = (int(nums.group()) * 60 if nums and 'hour' in low else int(nums.group()) if nums else 60)
            reply = 'Which style do you prefer: theory, practical exercises, or a mixture of both?'
        elif not s.get('learning_style'):
            s['learning_style'] = 'Theory' if 'theory' in low and 'mixture' not in low else ('Practical' if 'practical' in low and 'mixture' not in low else 'Mixed')
            s.setdefault('desired_outcome', 'Build practical skills and complete a final project'); s.setdefault('language','English')
            return {'state':s,'reply':self._profile_summary(s)+'\n\nYou can customize this profile or generate your course.','ready':True}
        else:
            if any(x in low for x in ['hour','minute']):
                nums=re.search(r'\d+',m)
                if nums: s['daily_minutes']=int(nums.group())*(60 if 'hour' in low else 1)
            if any(x in low for x in ['week','day']) and re.search(r'\d+',m):
                nums=re.search(r'\d+',m); s['duration_days']=int(nums.group())*(7 if 'week' in low else 1)
            if any(x in low for x in ['theory','practical','mixture']): s['learning_style']='Mixed' if 'mixture' in low else ('Theory' if 'theory' in low else 'Practical')
            return {'state':s,'reply':self._profile_summary(s)+'\n\nYou can keep customizing or choose Generate Course.','ready':self._is_ready(s)}
        return {'state':s,'reply':reply,'ready':self._is_ready(s)}

    @staticmethod
    def _is_ready(s): return all(s.get(k) for k in ['topic','skill_level','duration_days','daily_minutes','learning_style'])

    @staticmethod
    def _profile_summary(s):
        return (f"**Your course profile**\n- Topic: {s.get('topic','Not set')}\n- Goal: {s.get('learning_goal','Build practical skills')}\n- Level: {s.get('skill_level','Not set')}\n- Duration: {s.get('duration_days','?')} days\n- Daily study: {s.get('daily_minutes','?')} minutes\n- Style: {s.get('learning_style','Not set')}\n- Outcome: {s.get('desired_outcome','Build practical skills')}\n- Language: {s.get('language','English')}")

    def _sample_course(self, p):
        """Honest offline fallback: topic-specific starter course, clearly marked as sample."""
        topic = (p.get('topic') or 'General Learning').strip()
        low = topic.lower()
        duration = int(p.get('duration_days') or 30)
        level = p.get('skill_level') or 'Beginner'
        if any(k in low for k in ('web development','html','css','javascript','frontend','full stack')):
            themes = [('Web Platform Foundations','HTML document structure, semantic elements, forms, browser requests and developer tools.'),('CSS and Responsive Design','Selectors, box model, layout, Flexbox, Grid, responsive breakpoints and accessibility.'),('JavaScript and the DOM','Values, conditions, functions, arrays, objects, events, DOM updates and browser debugging.'),('APIs and Asynchronous JavaScript','HTTP methods, JSON, fetch, promises, async/await, loading states and error handling.'),('Testing and Deployment','Form validation, responsive checks, accessibility, Git basics and deployment workflow.'),('Portfolio Capstone','Build and deploy a responsive data-driven web application with README and testing checklist.')]
        elif any(k in low for k in ('java','dsa','data structure','algorithm','programming')):
            themes = [('Language and Problem-Solving Foundations','Variables, types, operators, input/output, control flow and tracing program execution.'),('Methods and Core Data Types','Methods, scope, arrays, strings, debugging and decomposition into small functions.'),('Object-Oriented Programming','Classes, objects, constructors, encapsulation, inheritance, polymorphism and composition.'),('Core Data Structures','Array operations, linked-list concepts, stacks, queues, hash maps and trade-offs.'),('Algorithms and Complexity','Linear and binary search, sorting, recursion, two pointers and Big-O reasoning.'),('Testing and Interview Capstone','Solve mixed problems, test edge cases, explain complexity and build a small Java project.')]
        elif any(k in low for k in ('machine learning','ml','artificial intelligence','ai')):
            themes = [('Data and Python Foundations','Python syntax, notebooks, NumPy, pandas, data types and reproducible workflows.'),('Data Cleaning and Exploration','Missing values, duplicates, distributions, visualization and leakage prevention.'),('Supervised Learning','Features, labels, train/test split, regression, classification and baseline models.'),('Evaluation and Model Selection','Cross-validation, accuracy, precision, recall, F1, confusion matrices and overfitting.'),('Feature Engineering and Pipelines','Preprocessing, scaling, categorical encoding, pipelines and model comparison.'),('Responsible ML Capstone','Build an end-to-end model, document limitations, check bias and present results.')]
        elif 'python' in low:
            themes = [('Python Fundamentals','Variables, built-in types, expressions, conditions, loops and input/output.'),('Functions and Collections','Functions, scope, lists, tuples, sets, dictionaries and comprehensions.'),('Files, Errors and Modules','Exceptions, file handling, imports, virtual environments and debugging.'),('Working with Data','CSV/JSON, data validation, transformations and basic analysis.'),('Testing and Code Quality','Small functions, assertions, unit-test concepts, naming and documentation.'),('Practical Python Capstone','Build a command-line application with persistent data, validation and tests.')]
        elif any(k in low for k in ('data science','data analytics','analytics')):
            themes = [('Data Questions and Data Types','Define a question, identify variables, distinguish numerical/categorical data and assess data quality.'),('Cleaning and Preparation','Handle missing values, duplicates, inconsistent formats and outliers with documented decisions.'),('Exploratory Data Analysis','Use summaries, distributions, grouping and comparisons to identify patterns.'),('Visualization and Communication','Choose chart types, label axes, avoid misleading scales and explain uncertainty.'),('Statistics and Interpretation','Samples, mean/median, variability, correlation versus causation and basic inference.'),('Data Storytelling Capstone','Analyze a dataset, build a clear report and support conclusions with evidence.')]
        else:
            themes = [(f'{topic}: Foundations','Define the field, its vocabulary, core concepts, common use cases and prerequisites.'),(f'{topic}: Core Principles','Study the main principles and how they relate to each other, using concrete examples.'),(f'{topic}: Methods and Tools','Compare common methods, select appropriate tools and follow a repeatable workflow.'),(f'{topic}: Guided Applications','Work through realistic scenarios, interpret results and troubleshoot common mistakes.'),(f'{topic}: Practice and Evaluation','Complete structured exercises, verify results and explain decisions.'),(f'{topic}: Final Project','Apply the concepts to a scoped project and present the result, evidence and limitations.')]
        modules=[]
        for i,(mtitle,desc) in enumerate(themes):
            lesson_specs=[('Understand the concepts',f'{desc} Start by defining each term in your own words. Identify what problem the concept solves, the assumptions it relies on, and when it is useful. Compare a correct use with a common mistake. Keep notes of questions that need verification.'),('Work through a concrete example',f'Apply the ideas from {mtitle.lower()} to a small, realistic case. State the input, describe each step, and check the result against an independent expectation. If the result is unexpected, isolate one assumption at a time rather than changing several things at once.'),('Independent practice and review',f'Use the same principles in a new case without copying the worked example. Record your approach, test edge cases, and explain why your result is reasonable. Review the explanation and correct any gap before continuing.')]
            lessons=[]
            for j,(ltitle,explanation) in enumerate(lesson_specs):
                lessons.append({'title':f'{mtitle}: {ltitle}','explanation':explanation,'concepts':[desc,'Assumptions and limitations','A method for checking your result'],'examples':[f'Worked case: choose a small example related to {topic}, write down the input and expected outcome, then apply the steps one by one.','Counterexample: change one important input or assumption and explain how the outcome should change.'],'exercises':[f'Create a one-page note explaining three key ideas in {mtitle.lower()} and when each is useful.','Complete a new example independently, then check it against a trusted reference or test case.'],'takeaways':[f'Explain the main ideas behind {mtitle.lower()} in your own words.','Verify results and document assumptions instead of guessing.']})
            qs=[{'prompt':f'What is the most reliable way to apply {mtitle.lower()}?', 'options':['State the goal, apply the relevant concepts, and verify the result','Skip checking because the first result is always correct','Memorize one example and use it for every situation','Ignore assumptions and edge cases'],'correct_index':0,'explanation':'A sound workflow starts with the goal and relevant concepts, then checks results and assumptions.'},{'prompt':f'Why should you test a second example in {mtitle.lower()}?', 'options':['To avoid understanding the topic','To see whether the method works beyond one case and expose edge cases','To make the notes longer','To replace reliable references'],'correct_index':1,'explanation':'A second case helps reveal hidden assumptions and checks whether understanding generalizes.'},{'prompt':f'Which study note is most useful after {mtitle.lower()}?', 'options':['A copied paragraph with no explanation','Only a list of unfamiliar terms','A short explanation, worked example, and limitations','An answer with no evidence'],'correct_index':2,'explanation':'Connecting definitions, examples, and limitations supports transfer to new problems.'},{'prompt':f'If your result conflicts with a trusted reference while studying {mtitle.lower()}, what should you do?', 'options':['Assume the reference is wrong','Delete the example','Change many steps at once','Check inputs, assumptions, and each step systematically'],'correct_index':3,'explanation':'Systematic debugging helps identify the exact source of the discrepancy.'}]
            modules.append({'title':mtitle,'description':desc,'duration_minutes':max(30,round(duration*int(p.get('daily_minutes') or 60)/60/6*60)),'objectives':[f'Explain the core concepts in {mtitle.lower()}',f'Apply the ideas to a concrete {topic} example'],'lessons':lessons,'questions':qs})
        return {'title':f'{topic}: {duration}-Day Learning Path','description':f'A {level.lower()}-level starter outline for {topic}, planned for {duration} days with {p.get("daily_minutes",60)} minutes per day. This is offline sample content, not Google-verified material. Add a Gemini API key and disable demo mode for grounded AI course generation.','objectives':[f'Build a reliable foundation in {topic}','Apply the main concepts through guided and independent practice','Evaluate results and explain decisions','Complete a practical final project'],'sources':[],'modules':modules}

    @staticmethod
    def _demo_tutor(question, lesson, mode):
        title = lesson.get('title', 'this topic') if lesson else 'this topic'
        return (f"I'm currently in offline Demo Mode, so I can't reliably answer a specific subject question or verify it against Google. "
                f"Your question is about **{title}**: “{question}”. Add a valid `GEMINI_API_KEY` and set `DEMO_MODE=false` in `.env` to enable Gemini and Google Search-grounded answers. "
                "I don't want to guess and give you an incorrect explanation.")
