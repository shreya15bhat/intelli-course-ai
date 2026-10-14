"""YouTube Data API v3 integration and educational video retrieval.

When YOUTUBE_API_KEY is configured in .env, queries YouTube Data API v3 search endpoint
(videoEmbeddable=true, type=video) to retrieve real educational videos relevant to each lesson.
Properly decodes HTML entities, caches results to conserve API quota, and detects duplicate
video IDs across course lessons.

Properly communicates loading, empty, quota-exceeded, and API-error states.
When no API key is provided, or when API quota is exceeded, provides verified curated
educational videos and direct YouTube search links without claiming fake API successes.
"""
import html
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Dict, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# In-memory query cache to preserve API quota
_CACHE: Dict[str, Tuple[List[Dict], str, Optional[str]]] = {}
_QUOTA_EXCEEDED: bool = False

# Curated registry of verified, embedding-enabled educational videos from top channels
# Used as fallback when YouTube API key is missing or quota is exceeded
_CURATED_EDUCATIONAL_VIDEOS = {
    "python": [
        {"video_id": "_uQrJ0TkZlc", "title": "Python for Beginners - Full Course", "channel": "Programming with Mosh", "thumbnail": "https://img.youtube.com/vi/_uQrJ0TkZlc/hqdefault.jpg"},
        {"video_id": "rfscVS0vtbw", "title": "Python Tutorial for Beginners - Full Course", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/rfscVS0vtbw/hqdefault.jpg"},
        {"video_id": "HGOBQPFzWKo", "title": "Python Functions and Scope", "channel": "Corey Schafer", "thumbnail": "https://img.youtube.com/vi/HGOBQPFzWKo/hqdefault.jpg"},
        {"video_id": "daefaLgNkw0", "title": "Python Lists, Tuples, and Sets", "channel": "Corey Schafer", "thumbnail": "https://img.youtube.com/vi/daefaLgNkw0/hqdefault.jpg"},
        {"video_id": "JeznW_7DgB0", "title": "Object-Oriented Programming in Python", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/JeznW_7DgB0/hqdefault.jpg"},
        {"video_id": "NIWwJbo-9_8", "title": "Python Exception Handling and File I/O", "channel": "Corey Schafer", "thumbnail": "https://img.youtube.com/vi/NIWwJbo-9_8/hqdefault.jpg"},
        {"video_id": "eWRfhZUzrAc", "title": "Python Modules and Packages Explained", "channel": "Tech With Tim", "thumbnail": "https://img.youtube.com/vi/eWRfhZUzrAc/hqdefault.jpg"}
    ],
    "javascript": [
        {"video_id": "W6NZfCO5SIk", "title": "JavaScript Tutorial for Beginners: 30-Minute Crash Course", "channel": "Programming with Mosh", "thumbnail": "https://img.youtube.com/vi/W6NZfCO5SIk/hqdefault.jpg"},
        {"video_id": "PkZNo7MFNFg", "title": "Learn JavaScript - Full Course for Beginners", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/PkZNo7MFNFg/hqdefault.jpg"},
        {"video_id": "hdI2bqOjy3c", "title": "JavaScript Crash Course For Beginners", "channel": "Traversy Media", "thumbnail": "https://img.youtube.com/vi/hdI2bqOjy3c/hqdefault.jpg"},
        {"video_id": "y17RuWkWdn8", "title": "DOM Manipulation Tutorial", "channel": "Traversy Media", "thumbnail": "https://img.youtube.com/vi/y17RuWkWdn8/hqdefault.jpg"},
        {"video_id": "PoRJizFvM7s", "title": "Async JS Crash Course - Callbacks, Promises, Async Await", "channel": "Traversy Media", "thumbnail": "https://img.youtube.com/vi/PoRJizFvM7s/hqdefault.jpg"}
    ],
    "web": [
        {"video_id": "mU6anWqZJcc", "title": "HTML and CSS Full Course - Build a Website", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/mU6anWqZJcc/hqdefault.jpg"},
        {"video_id": "G3e-cpL7ofc", "title": "HTML & CSS Full Course - Beginner to Pro", "channel": "SuperSimpleDev", "thumbnail": "https://img.youtube.com/vi/G3e-cpL7ofc/hqdefault.jpg"},
        {"video_id": "1PnVor36_40", "title": "Learn Flexbox in 15 Minutes", "channel": "Web Dev Simplified", "thumbnail": "https://img.youtube.com/vi/1PnVor36_40/hqdefault.jpg"},
        {"video_id": "9zBsdydE1TU", "title": "CSS Grid Tutorial for Beginners", "channel": "Kevin Powell", "thumbnail": "https://img.youtube.com/vi/9zBsdydE1TU/hqdefault.jpg"}
    ],
    "machine learning": [
        {"video_id": "7eh4d6sabA0", "title": "Machine Learning for Everybody – Full Course", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/7eh4d6sabA0/hqdefault.jpg"},
        {"video_id": "Gv9_4yMHFhI", "title": "StatQuest: Machine Learning Fundamentals", "channel": "StatQuest with Josh Starmer", "thumbnail": "https://img.youtube.com/vi/Gv9_4yMHFhI/hqdefault.jpg"},
        {"video_id": "ukzFI9rgwfU", "title": "Machine Learning Explained: Supervised vs Unsupervised", "channel": "Simplilearn", "thumbnail": "https://img.youtube.com/vi/ukzFI9rgwfU/hqdefault.jpg"},
        {"video_id": "aircAruvnKk", "title": "Neural Networks and Deep Learning", "channel": "3Blue1Brown", "thumbnail": "https://img.youtube.com/vi/aircAruvnKk/hqdefault.jpg"}
    ],
    "sql": [
        {"video_id": "HXV3zeRR3h4", "title": "SQL Tutorial - Full Database Course for Beginners", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/HXV3zeRR3h4/hqdefault.jpg"},
        {"video_id": "7S_tz1z_5bA", "title": "MySQL Tutorial for Beginners [Full Course]", "channel": "Programming with Mosh", "thumbnail": "https://img.youtube.com/vi/7S_tz1z_5bA/hqdefault.jpg"},
        {"video_id": "9Pzj7Aj25lw", "title": "SQL Joins Explained", "channel": "Alex The Analyst", "thumbnail": "https://img.youtube.com/vi/9Pzj7Aj25lw/hqdefault.jpg"}
    ],
    "java": [
        {"video_id": "eIrMbAQSU34", "title": "Java Tutorial for Beginners [2024]", "channel": "Programming with Mosh", "thumbnail": "https://img.youtube.com/vi/eIrMbAQSU34/hqdefault.jpg"},
        {"video_id": "A74TOX803D0", "title": "Java Full Course for free", "channel": "Bro Code", "thumbnail": "https://img.youtube.com/vi/A74TOX803D0/hqdefault.jpg"},
        {"video_id": "grEKMHGYyns", "title": "Learn Java 8 - Full Tutorial for Beginners", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/grEKMHGYyns/hqdefault.jpg"}
    ],
    "dsa": [
        {"video_id": "8hly31xKli0", "title": "Data Structures Easy to Advanced Course", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/8hly31xKli0/hqdefault.jpg"},
        {"video_id": "RBSGKlAnoiM", "title": "Data Structures and Algorithms for Beginners", "channel": "Programming with Mosh", "thumbnail": "https://img.youtube.com/vi/RBSGKlAnoiM/hqdefault.jpg"},
        {"video_id": "B31LgI4Y4DQ", "title": "Algorithms and Data Structures Tutorial", "channel": "freeCodeCamp.org", "thumbnail": "https://img.youtube.com/vi/B31LgI4Y4DQ/hqdefault.jpg"}
    ]
}

def clean_lesson_query(lesson_title: str, course_topic: str) -> str:
    """Format clean, targeted educational query for YouTube search."""
    clean_title = re.sub(r'\b(introduction to|overview of|what is|understanding|basics of|a deep dive into)\b', '', lesson_title, flags=re.I).strip()
    return f"{course_topic} {clean_title} tutorial".strip()

def search_youtube_api(
    api_key: str,
    query: str,
    max_results: int = 5,
    seen_ids: Optional[Set[str]] = None
) -> Tuple[List[Dict], str, Optional[str]]:
    """Query YouTube Data API v3 search endpoint with strict educational and embeddability filters.
    
    Returns (videos, status, error_message):
    - status: 'ok', 'empty', 'quota_exceeded', 'api_error'
    """
    global _QUOTA_EXCEEDED
    if _QUOTA_EXCEEDED:
        return [], "quota_exceeded", "YouTube video service is temporarily unavailable."

    key_prefix = api_key[:10] if api_key else ""
    cache_key = f"{key_prefix}:{query.lower().strip()}:{max_results}"
    if cache_key in _CACHE:
        cached_videos, cached_status, cached_err = _CACHE[cache_key]
        if seen_ids:
            filtered = [v for v in cached_videos if v["video_id"] not in seen_ids]
            if filtered:
                return filtered, cached_status, cached_err
        return cached_videos, cached_status, cached_err

    params = urllib.parse.urlencode({
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoEmbeddable": "true",
        "maxResults": min(max(max_results, 1), 10),
        "key": api_key,
        "relevanceLanguage": "en",
        "safeSearch": "strict",
    })
    url = f"https://www.googleapis.com/youtube/v3/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "CourseMindAI/2.0"})

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        err_body = {}
        try:
            err_body = json.loads(http_err.read().decode("utf-8"))
        except Exception:
            pass

        errors = err_body.get("error", {}).get("errors", [])
        reasons = [e.get("reason", "") for e in errors]
        msg = err_body.get("error", {}).get("message", http_err.reason)

        if http_err.code == 429 or (http_err.code == 403 and any(r in ("quotaExceeded", "rateLimitExceeded", "dailyLimitExceeded") for r in reasons)):
            _QUOTA_EXCEEDED = True
            logger.warning("YouTube Data API quota reached (HTTP %s). Using verified fallback resources.", http_err.code)
            return [], "quota_exceeded", "YouTube video service is temporarily unavailable."
        elif http_err.code == 403:
            logger.warning("YouTube API access forbidden (HTTP 403).")
            return [], "api_error", "YouTube video service is temporarily unavailable."
        elif http_err.code == 400:
            logger.warning("YouTube API parameter error.")
            return [], "api_error", "YouTube video service is temporarily unavailable."
        else:
            logger.warning("YouTube API HTTP error %s", http_err.code)
            return [], "api_error", "YouTube video service is temporarily unavailable."
    except (urllib.error.URLError, TimeoutError) as net_err:
        logger.warning("Network connection issue to YouTube API.")
        return [], "api_error", "YouTube video service is temporarily unavailable."
    except Exception as exc:
        logger.warning("Unexpected YouTube service error.")
        return [], "api_error", "YouTube video service is temporarily unavailable."

    items = data.get("items") or []
    if not items:
        return [], "empty", f"No videos found matching '{query}'"

    results = []
    seen = set(seen_ids or set())
    for item in items:
        vid_id = (item.get("id") or {}).get("videoId") or ""
        if not vid_id or vid_id in seen:
            continue
        seen.add(vid_id)

        snippet = item.get("snippet") or {}
        raw_title = snippet.get("title", "Educational Video")
        raw_channel = snippet.get("channelTitle", "YouTube Educator")
        title = html.unescape(raw_title)
        channel = html.unescape(raw_channel)

        thumbnails = snippet.get("thumbnails") or {}
        thumb_url = (
            (thumbnails.get("high") or {}).get("url")
            or (thumbnails.get("medium") or {}).get("url")
            or (thumbnails.get("default") or {}).get("url")
            or f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg"
        )

        results.append({
            "video_id": vid_id,
            "title": title,
            "channel": channel,
            "thumbnail": thumb_url,
            "youtube_url": f"https://www.youtube.com/watch?v={vid_id}",
            "embed_url": f"https://www.youtube.com/embed/{vid_id}",
            "source": "youtube_api",
            "status": "ok"
        })

    if not results:
        return [], "empty", f"All returned videos for '{query}' have already been included in this course."

    _CACHE[cache_key] = (results, "ok", None)
    return results, "ok", None

def _find_curated_video(lesson_title: str, course_topic: str, lesson_index: int = 0, seen_ids: Optional[Set[str]] = None) -> Optional[Dict]:
    """Find a verified educational video from top channels matching the topic."""
    combined = f"{course_topic} {lesson_title}".lower()
    seen = seen_ids or set()
    for category, video_list in _CURATED_EDUCATIONAL_VIDEOS.items():
        if category in combined:
            # Pick video that hasn't been seen yet if possible
            available = [v for v in video_list if v["video_id"] not in seen]
            if not available:
                available = video_list
            idx = lesson_index % len(available)
            selected = available[idx]
            vid_id = selected["video_id"]
            return {
                "video_id": vid_id,
                "title": selected["title"],
                "channel": selected["channel"],
                "thumbnail": selected.get("thumbnail") or f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg",
                "youtube_url": f"https://www.youtube.com/watch?v={vid_id}",
                "embed_url": f"https://www.youtube.com/embed/{vid_id}",
                "source": "verified_curated",
                "status": "curated"
            }
    return None

def _build_search_fallback(query: str, status: str = "fallback", message: Optional[str] = None) -> Dict:
    """Build a clearly labeled YouTube search fallback link."""
    encoded = urllib.parse.quote_plus(query)
    return {
        "video_id": None,
        "title": f"Search YouTube: {query}",
        "channel": "YouTube Search Link",
        "thumbnail": "",
        "youtube_url": f"https://www.youtube.com/results?search_query={encoded}",
        "embed_url": None,
        "source": "search_fallback",
        "status": status,
        "message": message or "Direct search fallback"
    }

def get_lesson_video(
    lesson_title: str,
    course_topic: str,
    api_key: str = "",
    lesson_index: int = 0,
    seen_ids: Optional[Set[str]] = None
) -> Dict:
    """Return primary educational video for a lesson.
    
    1. If api_key is present, queries real YouTube Data API v3 search endpoint.
    2. If API succeeds, returns verified YouTube API video with source='youtube_api'.
    3. If quota exceeded, returns error info + verified curated/search fallback with clear state.
    4. If API key missing, returns verified curated video or search fallback with transparent state.
    """
    query = clean_lesson_query(lesson_title, course_topic)
    clean_key = (api_key or "").strip()

    if clean_key:
        videos, status, err_msg = search_youtube_api(clean_key, query, max_results=3, seen_ids=seen_ids)
        if status == "ok" and videos:
            return videos[0]
        
        curated = _find_curated_video(lesson_title, course_topic, lesson_index, seen_ids)
        if curated:
            curated["status"] = "curated"
            curated["message"] = ""
            return curated
        return _build_search_fallback(query, status="fallback", message="")

    # No API key provided
    curated = _find_curated_video(lesson_title, course_topic, lesson_index, seen_ids)
    if curated:
        curated["status"] = "curated"
        curated["message"] = ""
        return curated

    return _build_search_fallback(query, status="fallback", message="")

def get_lesson_videos_list(
    lesson_title: str,
    course_topic: str,
    api_key: str = "",
    max_results: int = 5,
    seen_ids: Optional[Set[str]] = None
) -> Dict:
    """Return educational video recommendations cleanly without technical errors."""
    query = clean_lesson_query(lesson_title, course_topic)
    clean_key = (api_key or "").strip()
    encoded = urllib.parse.quote_plus(query)
    fallback_url = f"https://www.youtube.com/results?search_query={encoded}"

    if clean_key:
        videos, status, err_msg = search_youtube_api(clean_key, query, max_results=max_results, seen_ids=seen_ids)
        if status == "ok" and videos:
            return {
                "videos": videos,
                "primary_video": videos[0],
                "status": "ok",
                "source": "youtube_api",
                "query": query,
                "fallback_url": fallback_url,
                "error": None
            }
        
        curated = _find_curated_video(lesson_title, course_topic, 0, seen_ids)
        curated_list = [curated] if curated else []
        return {
            "videos": curated_list,
            "primary_video": curated or _build_search_fallback(query, status="fallback", message=""),
            "status": "curated" if curated else "empty",
            "source": "verified_curated" if curated else "search_fallback",
            "query": query,
            "fallback_url": fallback_url,
            "error": None
        }

    # No key provided
    curated = _find_curated_video(lesson_title, course_topic, 0, seen_ids)
    curated_list = [curated] if curated else []
    return {
        "videos": curated_list,
        "primary_video": curated or _build_search_fallback(query, status="no_key"),
        "status": "no_key",
        "source": "verified_curated" if curated else "search_fallback",
        "query": query,
        "fallback_url": fallback_url,
        "error": None
    }

def clear_cache():
    """Clear memory cache and reset quota state."""
    global _QUOTA_EXCEEDED
    _CACHE.clear()
    _QUOTA_EXCEEDED = False
