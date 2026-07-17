import os
import re
import sys
import time
import json
import uuid
import logging
import subprocess
import urllib.request
import urllib.parse
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QFileDialog, QComboBox, QCheckBox, QGroupBox, QGridLayout, QSplitter,
    QMessageBox, QProgressBar, QTextEdit, QListWidget, QListWidgetItem,
    QScrollArea, QHeaderView, QTableWidget, QTableWidgetItem, QSpinBox,
    QLineEdit, QDialog, QProgressDialog, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QFont, QDragEnterEvent, QDropEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from ui.widgets.drag_drop import DragDropWidget, get_video_thumbnail
from services.ffmpeg_service import get_ffmpeg_cmd, get_ffprobe_cmd, has_audio_stream, get_video_fps, get_audio_channels
from settings.manager import program_path, get_default_export_dir

import threading

logger = logging.getLogger("video_dubber")

_whisper_model_cache = {}
_whisper_model_lock = threading.Lock()

def is_whisper_model_valid(path, model_size):
    config_path = os.path.join(path, "config.json")
    model_bin_path = os.path.join(path, "model.bin")
    if not (os.path.exists(config_path) and os.path.exists(model_bin_path)):
        return False
    
    # Check minimum file size for model.bin to detect incomplete downloads
    min_sizes = {
        "tiny": 70 * 1024 * 1024,      # ~75 MB
        "tiny.en": 70 * 1024 * 1024,
        "base": 130 * 1024 * 1024,     # ~140 MB
        "base.en": 130 * 1024 * 1024,
        "small": 400 * 1024 * 1024,    # ~460 MB
        "small.en": 400 * 1024 * 1024,
        "medium": 1400 * 1024 * 1024,  # ~1.5 GB
        "medium.en": 1400 * 1024 * 1024,
        "large-v1": 2800 * 1024 * 1024, # ~3 GB
        "large-v2": 2800 * 1024 * 1024,
        "large-v3": 2800 * 1024 * 1024,
        "large": 2800 * 1024 * 1024,
    }
    
    try:
        size = os.path.getsize(model_bin_path)
        min_size = min_sizes.get(model_size, 50 * 1024 * 1024)
        if size < min_size:
            return False
    except Exception:
        return False
        
    return True

def load_whisper_model_safe(model_path_or_name, device, compute_type, local_files_only, device_index=0):
    from faster_whisper import WhisperModel
    try:
        if device == "cuda":
            return WhisperModel(model_path_or_name, device=device, device_index=device_index, compute_type=compute_type, local_files_only=local_files_only)
        else:
            return WhisperModel(model_path_or_name, device=device, compute_type=compute_type, local_files_only=local_files_only)
    except Exception as e:
        err_msg = str(e).lower()
        if "incomplete" in err_msg or "failed to read" in err_msg or "truncated" in err_msg or "corrupt" in err_msg:
            # Case 1: Local directory is corrupt
            if os.path.isdir(model_path_or_name):
                import shutil
                logger.warning(f"Detected corrupt/incomplete Whisper model directory at {model_path_or_name}: {e}. Deleting and attempting online download...")
                try:
                    shutil.rmtree(model_path_or_name)
                except Exception as rmtree_err:
                    logger.error(f"Failed to delete corrupt model directory: {rmtree_err}")
                model_name = os.path.basename(model_path_or_name)
            else:
                # Case 2: Hugging Face cache is corrupt
                model_name = model_path_or_name
                hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                if os.path.isdir(hf_cache_dir):
                    target_substr = f"faster-whisper-{model_name}".lower()
                    for entry in os.listdir(hf_cache_dir):
                        if entry.lower().startswith("models--") and target_substr in entry.lower():
                            full_entry_path = os.path.join(hf_cache_dir, entry)
                            if os.path.isdir(full_entry_path):
                                logger.warning(f"Detected corrupt Hugging Face Whisper model cache at {full_entry_path}. Deleting...")
                                try:
                                    import shutil
                                    shutil.rmtree(full_entry_path)
                                except Exception as rmtree_err:
                                    logger.error(f"Failed to delete Hugging Face cache directory: {rmtree_err}")
            
            # Attempt to download/load fresh from online HF repository
            try:
                logger.info(f"Attempting to download and load fresh Whisper model '{model_name}' from Hugging Face...")
                if device == "cuda":
                    return WhisperModel(model_name, device=device, device_index=device_index, compute_type=compute_type, local_files_only=False)
                else:
                    return WhisperModel(model_name, device=device, compute_type=compute_type, local_files_only=False)
            except Exception as dl_err:
                logger.error(f"Failed to download/load Whisper model '{model_name}' from online repository: {dl_err}")
                raise RuntimeError(
                    f"Whisper model '{model_name}' is corrupt or incomplete and could not be loaded. "
                    f"Please check your internet connection or verify the model files at '{model_path_or_name}'."
                ) from dl_err
        else:
            raise e

def get_cached_whisper_model(model_path_or_name, device, compute_type, local_files_only, device_index=0):
    global _whisper_model_cache
    key = (model_path_or_name, device, compute_type, local_files_only, device_index)
    with _whisper_model_lock:
        if key in _whisper_model_cache:
            return _whisper_model_cache[key], True
        
        model = load_whisper_model_safe(model_path_or_name, device, compute_type, local_files_only, device_index)
        _whisper_model_cache[key] = model
        return model, False

def clear_whisper_model_cache():
    global _whisper_model_cache
    with _whisper_model_lock:
        if not _whisper_model_cache:
            return
        
        logger.info("🧹 Clearing global Whisper model cache from memory...")
        try:
            keys = list(_whisper_model_cache.keys())
            for k in keys:
                del _whisper_model_cache[k]
            _whisper_model_cache.clear()
            
            import gc
            gc.collect()
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("✨ Whisper model cache cleared successfully.")
        except Exception as e:
            logger.debug(f"Failed to clear global Whisper model cache: {e}")

def get_safe_device(device_name: str) -> str:
    return device_name






def clean_brackets(text: str) -> str:
    """
    Removes bracket/parenthesis and their contents from text so they aren't processed by TTS.
    """
    if not text:
        return ""
    # Remove brackets and everything inside them
    cleaned = re.sub(r'[\(\[\{（【「［].*?[\)\]\}）】」』］]', ' ', text)
    # Clean up double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_emotion(text: str) -> tuple[str, str]:
    """
    Extracts emotion from brackets and returns (cleaned_text, emotion_detected).
    """
    if not text:
        return "", ""
        
    pattern = r'[\(\[\{（【「［](.*?)[\)\]\}）】」』］]'
    matches = re.findall(pattern, text)
    
    emotion = ""
    cleaned_text = re.sub(pattern, ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    if matches:
        for m in matches:
            m_lower = m.lower().strip()
            # English and Khmer mappings for emotions
            if any(k in m_lower for k in ("angry", "furious", "rage", "ខឹង", "មួរម៉ៅ")):
                emotion = "angry"
                break
            elif any(k in m_lower for k in ("sad", "cry", "sorrow", "melancholy", "កើតទុក្ខ", "យំ", "ព្រួយបារម្ភ")):
                emotion = "sad"
                break
            elif any(k in m_lower for k in ("happy", "joy", "cheerful", "សប្បាយ", "សប្បាយចិត្ត", "រីករាយ")):
                emotion = "happy"
                break
            elif any(k in m_lower for k in ("excited", "thrilled", "រំភើប")):
                emotion = "excited"
                break
            elif any(k in m_lower for k in ("whisper", "softly", "mumble", "ខ្សឹប", "តិចៗ")):
                emotion = "whisper"
                break
            elif any(k in m_lower for k in ("scared", "fear", "terrified", "afraid", "ភ័យ", "ខ្លាច")):
                emotion = "scared"
                break
            elif any(k in m_lower for k in ("surprised", "shocked", "amazed", "ភ្ញាក់ផ្អើល", "ស្រឡាំងកាំង")):
                emotion = "surprised"
                break
            elif any(k in m_lower for k in ("calm", "relaxed", "gentle", "ស្ងប់ស្ងាត់", "សុភាព")):
                emotion = "calm"
                break
            elif any(k in m_lower for k in ("thought", "mind", "monologue", "គិតក្នុងចិត្ត", "គិត")):
                emotion = "thought"
                break
                
    return cleaned_text, emotion

def save_resume_state(state):
    try:
        from settings.manager import program_path
        state_file = os.path.join(program_path, "settings", "resume_state.json")
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save resume state: {e}")

def load_resume_state():
    try:
        from settings.manager import program_path
        state_file = os.path.join(program_path, "settings", "resume_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load resume state: {e}")
    return None

def clear_resume_state():
    try:
        from settings.manager import program_path
        state_file = os.path.join(program_path, "settings", "resume_state.json")
        if os.path.exists(state_file):
            os.remove(state_file)
    except Exception as e:
        logger.error(f"Failed to clear resume state: {e}")

def load_dub_cache() -> dict:
    try:
        from settings.manager import program_path
        cache_file = os.path.join(program_path, "settings", "dub_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dub cache: {e}")
    return {}

def save_dub_cache(cache_data: dict):
    try:
        from settings.manager import program_path
        cache_file = os.path.join(program_path, "settings", "dub_cache.json")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Failed to save dub cache: {e}")

_GLOSSARY_CACHE = None

def load_translation_glossary(force_reload=False) -> tuple:
    """
    Loads translation glossary rules from translation_glossary.txt.
    Returns (pre_rules, post_rules).
    """
    global _GLOSSARY_CACHE
    if _GLOSSARY_CACHE is not None and not force_reload:
        return _GLOSSARY_CACHE

    pre_rules = {}
    post_rules = {}
    
    # Save/load from project root (parent directory of ui folder)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    glossary_path = os.path.join(base_path, "translation_glossary.txt")
    
    if not os.path.exists(glossary_path):
        default_content = """# ===================================================================
# Nova Translation Glossary (វចនានុក្រមបកប្រែ)
# Format: <Source Word> -> <Target Word>
# Rules with non-Khmer keys are applied BEFORE translation (source).
# Rules with Khmer keys are applied AFTER translation (target).
# Lines starting with '#' or empty lines are ignored.
# ===================================================================

# --- Pre-Translation Replacements (Source Guides) ---
主帅 -> 将军
别来无恙 -> 别来无恙
Young Master -> អ្នកប្រុស
young master -> អ្នកប្រុស
Old Master -> លោកម្ចាស់
old master -> លោកម្ចាស់
Commander -> មេបញ្ជាការ
commander -> មេបញ្ជាការ
General -> មេទ័ព
general -> មេទ័ព
Mr. President -> លោកប្រធាន
President -> លោកប្រធាន

# --- Post-Translation Replacements (Khmer Corrections) ---
គ្រូបង្វឹក -> មេទ័ព
លោកគ្រូវ័យក្មេង -> អ្នកប្រុស
ចៅហ្វាយវ័យក្មេង -> អ្នកប្រុស
ម្ចាស់វ័យក្មេង -> អ្នកប្រុស
លោកប្រធានាធិបតី -> លោកប្រធាន
អ្នកចាញ់ -> មនុស្សឥតប្រយោជន៍
ទាហានចាស់ -> ទាហានជើងចាស់
"""
        try:
            with open(glossary_path, "w", encoding="utf-8") as f:
                f.write(default_content)
        except Exception:
            pass

    if os.path.exists(glossary_path):
        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "->" in line:
                        parts = line.split("->")
                        if len(parts) == 2:
                            src = parts[0].strip()
                            tgt = parts[1].strip()
                            if src and tgt:
                                is_khmer = any(u'\u1780' <= c <= u'\u17ff' for c in tgt)
                                if is_khmer:
                                    post_rules[src] = tgt
                                else:
                                    pre_rules[src] = tgt
        except Exception as e:
            logger.error(f"Failed to read translation glossary: {e}")

    _GLOSSARY_CACHE = (pre_rules, post_rules)
    return _GLOSSARY_CACHE

def get_gemini_keys():
    """
    Retrieves the configured gemini_api_key from settings,
    and parses it into a list of keys split by commas, semicolons, newlines, or spaces.
    """
    gemini_key = ""
    try:
        from app.downloader import get_settings
        gemini_key = get_settings().get("gemini_api_key", "").strip()
    except Exception:
        pass
        
    if not gemini_key:
        try:
            from settings.manager import load_settings
            gemini_key = load_settings().get("gemini_api_key", "").strip()
        except Exception:
            pass
            
    keys = []
    if gemini_key:
        import re
        parts = re.split(r'[\s,;\n]+', gemini_key)
        keys = [p.strip() for p in parts if p.strip()]
    return keys

def translate_via_gemini(text: str, target_lang: str, source_lang: str, api_key: str) -> str:
    """Helper to translate a single text using Gemini API."""
    import json
    import urllib.request
    
    lang_names = {
        "km": "Khmer",
        "en": "English",
        "th": "Thai",
        "vi": "Vietnamese",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "es": "Spanish"
    }
    
    tgt_name = lang_names.get(target_lang, target_lang)
    src_name = lang_names.get(source_lang, "Auto Detect")
    if src_name == "auto":
        src_name = "Auto Detect"
    
    prompt = (
        f"Translate the following text from {src_name} to {tgt_name}. "
        "Keep the translation natural, fluent, and contextual. "
        "Only output the translated text. Do not add any explanation or preamble. "
        f"Text to translate:\n{text}"
    )
    
    model_name = "gemini-2.5-flash"
    try:
        from settings.manager import load_settings
        model_name = load_settings().get("gemini_model", "gemini-2.5-flash")
    except Exception:
        pass
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidate = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return candidate.strip()
    except Exception as e:
        logger.error(f"Gemini translation failed: {e}")
        return ""

def translate_texts_batch_gemini(texts: list, target_lang: str, source_lang: str, api_key: str) -> list:
    """Helper to translate a list of texts in batch using Gemini API with structured JSON output configuration."""
    import json
    import urllib.request
    
    lang_names = {
        "km": "Khmer",
        "en": "English",
        "th": "Thai",
        "vi": "Vietnamese",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "es": "Spanish"
    }
    
    tgt_name = lang_names.get(target_lang, target_lang)
    src_name = lang_names.get(source_lang, "Auto Detect")
    if src_name == "auto":
        src_name = "Auto Detect"
    
    payload_texts = json.dumps(texts, ensure_ascii=False)
    
    prompt = (
        f"You are a professional translator translating from {src_name} to {tgt_name}. "
        "Translate each item in the following JSON array. "
        "Respond ONLY with a valid JSON array containing the translated items in the exact same order. "
        "Do not wrap the response in markdown code blocks or add any other text. "
        f"JSON Array:\n{payload_texts}"
    )
    
    model_name = "gemini-2.5-flash"
    try:
        from settings.manager import load_settings
        model_name = load_settings().get("gemini_model", "gemini-2.5-flash")
    except Exception:
        pass
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            reply = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            translated_list = json.loads(reply)
            if isinstance(translated_list, list) and len(translated_list) == len(texts):
                return translated_list
    except Exception as e:
        logger.error(f"Gemini batch translation failed: {e}")
        
    return []

def translate_text(text: str, target_lang: str = "km", source_lang: str = "auto") -> str:
    """Translates text using Gemini translation (if configured) or free Google Translate API fallback."""
    if not text.strip():
        return ""
        
    pre_rules, post_rules = load_translation_glossary()
    
    # 0. Apply automatic regex guides for English titles (e.g. "Miss Su" -> "Ms. Su")
    if target_lang == "km":
        import re
        text = re.sub(r'\bMiss\s+([A-Z][a-zA-Z]*)', r'Ms. \1', text)
    
    # 1. Apply pre-translation replacements
    for src, tgt in pre_rules.items():
        text = text.replace(src, tgt)
    
    translated = ""
    engine = "google"
    try:
        from app.downloader import get_settings
        engine = get_settings().get("translation_engine", "")
    except Exception:
        pass
        
    if not engine:
        try:
            from settings.manager import load_settings
            settings = load_settings()
            has_key = bool(settings.get("gemini_api_key", "").strip())
            default_engine = "gemini" if has_key else "google"
            engine = settings.get("translation_engine", default_engine)
        except Exception:
            engine = "google"

    if engine == "gemini":
        keys = get_gemini_keys()
        if keys:
            import random
            start_idx = random.randint(0, len(keys) - 1)
            for attempt in range(len(keys)):
                current_key = keys[(start_idx + attempt) % len(keys)]
                translated = translate_via_gemini(text, target_lang, source_lang, current_key)
                if translated:
                    break
        
    if not translated:
        # Fallback to Google Translate free API
        # Retry logic to handle rate-limiting or network issues
        for attempt in range(3):
            try:
                url = "https://translate.googleapis.com/translate_a/single"
                params = {
                    "client": "gtx",
                    "sl": source_lang,
                    "tl": target_lang,
                    "dt": "t",
                    "q": text
                }
                query_string = urllib.parse.urlencode(params)
                req = urllib.request.Request(f"{url}?{query_string}", headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'
                })
                with urllib.request.urlopen(req, timeout=8) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    translated = "".join([part[0] for part in res[0] if part[0]])
                    if translated:
                        break
            except Exception as e:
                logger.warning(f"Translation attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(1.0)
                    
    if not translated:
        # If translation fails completely and the original text is Chinese,
        # returning Chinese text to a Khmer TTS engine will result in gibberish.
        has_chinese = any(u'\u4e00' <= char <= u'\u9fff' for char in text)
        if has_chinese and target_lang == "km":
            return ""
        translated = text
        
    # 2. Apply post-translation replacements (only if target is Khmer)
    if target_lang == "km" and translated:
        for src, tgt in post_rules.items():
            translated = translated.replace(src, tgt)
            
    return translated

def translate_texts_batch(texts: list, target_lang: str = "km", source_lang: str = "auto") -> list:
    """
    Translates a list of texts in batch using Google Translate API with glossary support.
    To avoid rate-limiting and improve performance, it sends text joined by newlines.
    Falls back to translating individually or in smaller chunks if the result count doesn't match.
    """
    if not texts:
        return []
        
    pre_rules, post_rules = load_translation_glossary(force_reload=True)
    
    # 1. Apply pre-translation rules to the source texts
    processed_texts = []
    for text in texts:
        if target_lang == "km":
            import re
            text = re.sub(r'\bMiss\s+([A-Z][a-zA-Z]*)', r'Ms. \1', text)
            
        for src, tgt in pre_rules.items():
            text = text.replace(src, tgt)
        processed_texts.append(text)
        
    engine = "google"
    try:
        from app.downloader import get_settings
        engine = get_settings().get("translation_engine", "")
    except Exception:
        pass
        
    if not engine:
        try:
            from settings.manager import load_settings
            settings = load_settings()
            has_key = bool(settings.get("gemini_api_key", "").strip())
            default_engine = "gemini" if has_key else "google"
            engine = settings.get("translation_engine", default_engine)
        except Exception:
            engine = "google"

    if engine == "gemini":
        keys = get_gemini_keys()
        if keys:
            logger.info(f"🌐 Translating in batch using Gemini API (Keys configured: {len(keys)})...")
            non_empty_indices = [idx for idx, t in enumerate(processed_texts) if t.strip()]
            if not non_empty_indices:
                return ["" if t.strip() == "" else t for t in processed_texts]
                
            non_empty_texts = [processed_texts[idx] for idx in non_empty_indices]
            
            gemini_results_map = {}
            gemini_success = True
            
            # Process in chunks of 40 to avoid hitting API rate limits or payload sizes
            for i in range(0, len(non_empty_texts), 40):
                chunk_texts = non_empty_texts[i : i + 40]
                chunk_res = []
                
                import random
                import time
                start_idx = random.randint(0, len(keys) - 1)
                for attempt in range(len(keys)):
                    current_key = keys[(start_idx + attempt) % len(keys)]
                    for sub_attempt in range(3):
                        chunk_res = translate_texts_batch_gemini(chunk_texts, target_lang=target_lang, source_lang=source_lang, api_key=current_key)
                        if chunk_res and len(chunk_res) == len(chunk_texts):
                            break
                        if sub_attempt < 2:
                            logger.warning(f"Gemini chunk translation attempt {sub_attempt + 1} failed. Retrying in 2 seconds...")
                            time.sleep(2)
                    if chunk_res and len(chunk_res) == len(chunk_texts):
                        break
                if chunk_res and len(chunk_res) == len(chunk_texts):
                    for idx_in_chunk, val in enumerate(chunk_res):
                        real_idx = non_empty_indices[i + idx_in_chunk]
                        gemini_results_map[real_idx] = val
                else:
                    gemini_success = False
                    break
                    
            if gemini_success and len(gemini_results_map) == len(non_empty_texts):
                results = []
                for idx, text in enumerate(processed_texts):
                    if idx in gemini_results_map:
                        res = gemini_results_map[idx]
                    else:
                        res = ""
                        
                    if target_lang == "km" and res:
                        for src, tgt in post_rules.items():
                            res = res.replace(src, tgt)
                    results.append(res)
                return results
            else:
                logger.warning("Gemini batch translation failed or returned incorrect length. Falling back to Google Translate...")

    results = [None] * len(processed_texts)
    # Filter out empty texts but keep track of indices
    non_empty_indices = [i for i, t in enumerate(processed_texts) if t.strip()]
    if not non_empty_indices:
        return ["" if t.strip() == "" else t for t in processed_texts]
        
    non_empty_texts = [processed_texts[i] for i in non_empty_indices]
    
    # We can group non_empty_texts into chunks where the total character count is <= 2500
    chunks = []
    current_chunk = []
    current_len = 0
    for text in non_empty_texts:
        if current_len + len(text) + 1 > 2500:
            chunks.append(current_chunk)
            current_chunk = [text]
            current_len = len(text)
        else:
            current_chunk.append(text)
            current_len += len(text) + 1
    if current_chunk:
        chunks.append(current_chunk)
        
    # Process each chunk
    chunk_start_idx = 0
    for chunk in chunks:
        chunk_indices = non_empty_indices[chunk_start_idx : chunk_start_idx + len(chunk)]
        chunk_start_idx += len(chunk)
        
        joined_text = "\n".join(chunk)
        translated_chunk_lines = []
        success = False
        
        for attempt in range(3):
            try:
                url = "https://translate.googleapis.com/translate_a/single"
                params = {
                    "client": "gtx",
                    "sl": source_lang,
                    "tl": target_lang,
                    "dt": "t",
                    "q": joined_text
                }
                query_string = urllib.parse.urlencode(params)
                req = urllib.request.Request(f"{url}?{query_string}", headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'
                })
                with urllib.request.urlopen(req, timeout=10) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    if res and res[0]:
                        parts = res[0]
                        translated_lines = []
                        for part in parts:
                            if part and part[0]:
                                translated_lines.append(part[0])
                        
                        full_translated = "".join(translated_lines)
                        lines = [line.strip() for line in full_translated.split('\n')]
                        if len(lines) > len(chunk) and lines[-1] == "":
                            lines = lines[:-1]
                            
                        if len(lines) == len(chunk):
                            # Validate each line for correct target language encoding (avoid incorrect local language fallback)
                            corrected_lines = []
                            for orig_line, trans_line in zip(chunk, lines):
                                has_khmer = any(u'\u1780' <= c <= u'\u17ff' for c in trans_line)
                                has_chinese = any(u'\u4e00' <= c <= u'\u9fff' for c in trans_line)
                                
                                # Check 1: Target is not Khmer, but result contains Khmer
                                if target_lang != "km" and has_khmer:
                                    logger.warning(f"Unexpected Khmer in '{target_lang}' translation. Re-translating line: '{orig_line}'")
                                    corrected_line = translate_text(orig_line, target_lang=target_lang, source_lang=source_lang)
                                    corrected_lines.append(corrected_line)
                                # Check 2: Target is Khmer, but result contains Chinese
                                elif target_lang == "km" and has_chinese:
                                    logger.warning(f"Untranslated Chinese in Khmer translation. Re-translating line: '{orig_line}'")
                                    corrected_line = translate_text(orig_line, target_lang=target_lang, source_lang=source_lang)
                                    corrected_lines.append(corrected_line)
                                else:
                                    corrected_lines.append(trans_line)
                                    
                            translated_chunk_lines = corrected_lines
                            success = True
                            break
                        else:
                            logger.warning(f"Batch translation returned {len(lines)} lines but expected {len(chunk)} lines.")
            except Exception as e:
                logger.warning(f"Batch translation attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(1.0)
                    
        if success:
            for idx, line in zip(chunk_indices, translated_chunk_lines):
                results[idx] = line
        else:
            from concurrent.futures import ThreadPoolExecutor
            def translate_single(text):
                return translate_text(text, target_lang=target_lang, source_lang=source_lang)
                
            with ThreadPoolExecutor(max_workers=5) as executor:
                translated_individual = list(executor.map(translate_single, chunk))
            for idx, trans in zip(chunk_indices, translated_individual):
                results[idx] = trans

    for i in range(len(processed_texts)):
        if results[i] is None:
            results[i] = ""
            
    # 2. Apply post-translation rules to the translated results (only if target is Khmer)
    final_results = []
    for res in results:
        if target_lang == "km" and res:
            for src, tgt in post_rules.items():
                res = res.replace(src, tgt)
        final_results.append(res)
            
    return final_results

def generate_voxcpm_tts(text: str, output_path: str, control_instruction: str, reference_wav: str = None) -> bool:
    """Generates speech using OpenBMB VoxCPM Hugging Face Space API with multi-endpoint fallback."""
    import requests
    import json
    import os
    
    endpoints = [
        "https://openbmb-voxcpm-demo.hf.space"
    ]
    
    api_name = "generate"
    
    # Startupinfo for subprocess
    startupinfo = None
    if os.name == 'nt':
        import subprocess
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        
    for base_url in endpoints:
        for attempt in range(3):
            logger.info(f"Trying VoxCPM endpoint: {base_url} (attempt {attempt+1}/3)...")
            
            ref_audio_filedata = None
            if reference_wav and os.path.exists(reference_wav):
                try:
                    # Upload the reference audio file to the Gradio Space
                    upload_url = f"{base_url}/gradio_api/upload"
                    fn = os.path.basename(reference_wav)
                    files = {"files": (fn, open(reference_wav, "rb"), "audio/wav")}
                    logger.info(f"Uploading custom voice reference audio to {base_url}: {fn}...")
                    upload_resp = requests.post(upload_url, files=files, timeout=30)
                    if upload_resp.status_code == 200:
                        remote_paths = upload_resp.json()
                        if remote_paths:
                            ref_audio_filedata = {
                                "path": remote_paths[0],
                                "meta": {"_type": "gradio.FileData"}
                            }
                            logger.info(f"Custom voice uploaded successfully to {base_url}. Remote path: {remote_paths[0]}")
                    else:
                        logger.error(f"Failed to upload reference audio to {base_url}. Status: {upload_resp.status_code}")
                        if attempt < 2:
                            import time
                            time.sleep(2)
                            continue
                except Exception as ue:
                    logger.error(f"Exception during reference audio upload to {base_url}: {ue}")
                    if attempt < 2:
                        import time
                        time.sleep(2)
                        continue
                    
            # Resolve parameter schema from API info
            param_count = 8
            try:
                info_resp = requests.get(f"{base_url}/gradio_api/info", timeout=10)
                if info_resp.status_code == 200:
                    gen_info = info_resp.json().get("named_endpoints", {}).get("/generate", {})
                    param_count = len(gen_info.get("parameters", []))
                    logger.info(f"Endpoint {base_url} supports {param_count} parameters.")
            except Exception as ie:
                logger.warning(f"Failed to fetch parameter count for {base_url}: {ie}. Defaulting to 8.")
                
            if param_count == 9:
                payload = {
                    "data": [
                        text,                  # target_text
                        control_instruction,   # control_instruction
                        ref_audio_filedata,    # reference_wav
                        False,                 # show_prompt_text
                        "",                    # prompt_text
                        2.0,                   # cfg_value
                        True,                  # DoNormalizeText
                        False,                 # DoDenoisePromptAudio
                        10                     # dit_steps
                    ]
                }
            else:
                payload = {
                    "data": [
                        text,                  # target_text
                        control_instruction,   # control_instruction
                        ref_audio_filedata,    # reference_wav
                        False,                 # show_prompt_text
                        "",                    # prompt_text
                        2.0,                   # cfg_value
                        True,                  # DoNormalizeText
                        False                  # DoDenoisePromptAudio
                    ]
                }
                
            headers = {"Content-Type": "application/json"}
            
            try:
                resp = requests.post(f"{base_url}/gradio_api/call/{api_name}", json=payload, headers=headers, timeout=20)
                if resp.status_code != 200:
                    logger.error(f"VoxCPM endpoint {base_url} API call failed with status code {resp.status_code}: {resp.text}")
                    if attempt < 2:
                        import time
                        time.sleep(2)
                        continue
                    continue
                    
                res_json = resp.json()
                event_id = res_json.get("event_id")
                if not event_id:
                    logger.error(f"VoxCPM endpoint {base_url} did not return an event_id")
                    if attempt < 2:
                        import time
                        time.sleep(2)
                        continue
                    continue
                    
                status_url = f"{base_url}/gradio_api/call/{api_name}/{event_id}"
                
                # Read the event-stream (SSE)
                r = requests.get(status_url, stream=True, timeout=60)
                current_event = None
                for line in r.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode('utf-8')
                    if line_str.startswith("event: "):
                        current_event = line_str[7:].strip()
                    elif line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if current_event == "complete":
                            data_json = json.loads(data_content)
                            if isinstance(data_json, list) and len(data_json) > 0:
                                audio_info = data_json[0]
                                if isinstance(audio_info, dict) and "path" in audio_info:
                                    file_path = audio_info["path"]
                                    file_url = audio_info.get("url") or f"{base_url}/gradio_api/file={file_path}"
                                    
                                    # Download the audio file
                                    audio_resp = requests.get(file_url, timeout=30)
                                    if audio_resp.status_code == 200:
                                        with open(output_path, "wb") as f:
                                            f.write(audio_resp.content)
                                        logger.info(f"Successfully generated VoxCPM speech from {base_url}")
                                        return True
                                    else:
                                        logger.error(f"Failed to download audio from {base_url}: {audio_resp.status_code}")
                        elif current_event == "error":
                            logger.error(f"VoxCPM endpoint {base_url} generation error: {data_content}")
                            break
                if attempt < 2:
                    import time
                    time.sleep(2)
                    continue
            except Exception as e:
                logger.error(f"Exception during VoxCPM TTS on {base_url}: {e}")
                if attempt < 2:
                    import time
                    time.sleep(2)
                    continue
            
    logger.critical("All VoxCPM endpoints failed.")
    return False

def resolve_auto_cloned_voice(detected_gender: str, saved_voices: list, default_path: str, ffmpeg_path: str, startupinfo=None) -> str:
    """
    Resolves the best matching cloned voice path for the given gender (male/female)
    by analyzing the pitch of the reference WAV files, falling back to name keywords.
    """
    import numpy as np
    selected_clone_path = default_path
    male_clones = []
    female_clones = []
    
    for v in saved_voices:
        name_lower = v["name"].lower()
        path_val = v["path"]
        
        detected_ref_gender = None
        if os.path.exists(path_val):
            try:
                import tempfile
                import wave
                temp_wav = os.path.join(tempfile.gettempdir(), f"temp_ref_{os.path.basename(path_val)}.wav")
                if os.path.exists(temp_wav):
                    try: os.remove(temp_wav)
                    except Exception: pass
                
                cmd_ref = [
                    ffmpeg_path, "-y", "-i", path_val,
                    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", "-t", "5", temp_wav
                ]
                import subprocess
                subprocess.run(cmd_ref, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
                
                if os.path.exists(temp_wav):
                    with wave.open(temp_wav, "rb") as w_ref:
                        ref_frames = w_ref.readframes(w_ref.getnframes())
                        ref_samples = np.frombuffer(ref_frames, dtype=np.int16)
                        ref_pitch = detect_pitch(ref_samples, sample_rate=24000)
                        if ref_pitch:
                            if ref_pitch < 175.0:
                                detected_ref_gender = "male"
                            else:
                                detected_ref_gender = "female"
                    try: os.remove(temp_wav)
                    except Exception: pass
            except Exception:
                pass
                
        if not detected_ref_gender:
            if any(x in name_lower for x in ("male", "ប្រុស", "man", "boy", "guy")):
                detected_ref_gender = "male"
            elif any(x in name_lower for x in ("female", "ស្រី", "woman", "girl", "lady")):
                detected_ref_gender = "female"
                
        if detected_ref_gender == "male":
            male_clones.append(path_val)
        elif detected_ref_gender == "female":
            female_clones.append(path_val)
            
    if detected_gender == "male":
        if male_clones:
            selected_clone_path = male_clones[0]
        elif saved_voices:
            selected_clone_path = saved_voices[min(1, len(saved_voices)-1)]["path"]
    else:
        if female_clones:
            selected_clone_path = female_clones[0]
        elif saved_voices:
            selected_clone_path = saved_voices[0]["path"]
            
    return selected_clone_path

def generate_local_voxcpm_tts(text: str, output_path: str, control_instruction: str, reference_wav: str = None, host="127.0.0.1", port=8000) -> bool:
    """Generates speech by calling the local VoxCPM API server."""
    import requests
    import base64
    import os
    
    url = f"http://{host}:{port}/generate"
    
    ref_b64 = None
    if reference_wav and os.path.exists(reference_wav):
        try:
            with open(reference_wav, "rb") as f:
                ref_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read local reference wav: {e}")
            
    from settings.manager import load_settings
    settings = load_settings()
    cfg_value = float(settings.get("voxcpm_cfg_value", 2.0))
    inference_timesteps = int(settings.get("voxcpm_inference_timesteps", 10))
    
    # Calculate a stable seed based on the reference wav path so the voice style remains consistent
    seed = 42
    if reference_wav:
        import hashlib
        seed = int(hashlib.md5(reference_wav.encode()).hexdigest(), 16) % 1000000
        
    payload = {
        "text": text,
        "control_instruction": control_instruction,
        "reference_wav_b64": ref_b64,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "seed": seed
    }
    
    import time
    for attempt in range(3):
        try:
            # Check health first
            health_resp = requests.get(f"http://{host}:{port}/health", timeout=2)
            if health_resp.status_code == 200:
                data = health_resp.json()
                if not data.get("model_loaded", False):
                    logger.warning(f"Local VoxCPM server is active but model weights are still loading (attempt {attempt+1}/3)...")
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    return False
            else:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return False
                
            resp = requests.post(url, json=payload, timeout=300)
            if resp.status_code == 200:
                res_data = resp.json()
                audio_b64 = res_data.get("audio_b64")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)
                    logger.info("Successfully generated local VoxCPM speech.")
                    return True
            logger.error(f"Local VoxCPM API failed (attempt {attempt+1}/3): {resp.status_code} - {resp.text}")
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            logger.debug(f"Local VoxCPM attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2)
                
    return False

def generate_tts_file(text: str, output_path: str, lang: str = "km", voice: str = None, custom_voice_path: str = None, auto_emotion: bool = False) -> bool:
    """Generates TTS audio file using edge-tts (or falls back to Google Translate TTS)."""
    if not text.strip():
        return False
        
    if voice and voice.startswith("voxcpm-custom|"):
        parts = voice.split("|", 1)
        voice = parts[0]
        custom_voice_path = parts[1]
        
    if not auto_emotion:
        clean_text = clean_brackets(text)
        if not clean_text:
            clean_text = text
        emotion = ""
    else:
        # Extract emotion from text and clean the text of bracketed contents
        clean_text, emotion = extract_emotion(text)
        if not clean_text:
            clean_text = text
        
    # Check if VoxCPM is selected
    if voice and "voxcpm" in voice:
        try:
            # Load local server settings
            from settings.manager import load_settings
            settings = load_settings()
            host = settings.get("voxcpm_host", "127.0.0.1")
            port = int(settings.get("voxcpm_port", 8000))
            backend = settings.get("voxcpm_backend", "local")
            success = False
            
            if backend == "local":
                # Auto-start local VoxCPM server if offline or model weights not loaded
                try:
                    from app.voxcpm_manager import LocalVoxCPMServerManager
                    manager = LocalVoxCPMServerManager()
                    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "VoxCPM2")
                    model_dir = settings.get("voxcpm_model_dir", default_dir)
                    device = get_safe_device(settings.get("voxcpm_device", "cuda"))
                    manager.configure(host, port, device, model_dir)
                    
                    running, loaded, _ = manager.is_running()
                    if not running:
                        logger.info("Local VoxCPM server is offline. Starting server automatically...")
                        started, msg = manager.start_server()
                        if not started:
                            logger.error(f"Failed to auto-start local VoxCPM server: {msg}")
                        else:
                            logger.info("Local VoxCPM server started. Waiting for model weights to load...")
                            import time
                            start_wait = time.time()
                            while time.time() - start_wait < 150:
                                is_run, is_loaded, error_msg = manager.is_running()
                                if error_msg:
                                    logger.error(f"Local VoxCPM server failed to load model weights: {error_msg}")
                                    break
                                if is_run and is_loaded:
                                    logger.info("Local VoxCPM model loaded successfully via auto-start.")
                                    break
                                time.sleep(2)
                            else:
                                logger.warning("Timed out waiting for local VoxCPM model weights to load.")
                    elif not loaded:
                        logger.info("Local VoxCPM server is active but model weights are still loading. Waiting...")
                        import time
                        start_wait = time.time()
                        while time.time() - start_wait < 150:
                            is_run, is_loaded, error_msg = manager.is_running()
                            if error_msg:
                                logger.error(f"Local VoxCPM model weights loading failed: {error_msg}")
                                break
                            if is_run and is_loaded:
                                logger.info("Local VoxCPM model loaded successfully.")
                                break
                            time.sleep(2)
                        else:
                            logger.warning("Timed out waiting for local VoxCPM model weights to load.")
                except Exception as auto_start_err:
                    logger.error(f"Error during local VoxCPM auto-start check: {auto_start_err}")

                # Call local generator
                if voice == "voxcpm-custom":
                    control_instruction = ""
                    success = generate_local_voxcpm_tts(
                        clean_text, output_path, control_instruction, 
                        reference_wav=custom_voice_path, host=host, port=port
                    )
                else:
                    if "female" in voice:
                        base_desc = "A young woman with a sweet, clear and gentle voice"
                        fallback_voice = "km-KH-SreymomNeural"
                    elif "male" in voice:
                        base_desc = "A mature man with a clear, calm and professional voice"
                        fallback_voice = "km-KH-PisethNeural"
                    else:
                        base_desc = "A speaker with a clear and natural voice"
                        fallback_voice = "km-KH-SreymomNeural"
                        
                    if not emotion:
                        if "-angry" in voice: emotion = "angry"
                        elif "-sad" in voice: emotion = "sad"
                        elif "-happy" in voice: emotion = "happy"
                        elif "-excited" in voice: emotion = "excited"
                        elif "-whisper" in voice: emotion = "whisper"
                        elif "-scared" in voice: emotion = "scared"
                        elif "-surprised" in voice: emotion = "surprised"
                        elif "-calm" in voice: emotion = "calm"
                        elif "-thought" in voice: emotion = "thought"

                    if voice in ("voxcpm-female", "voxcpm-male"):
                        emotion = ""

                    if emotion == "angry":
                        control_instruction = f"{base_desc}, speaking with an angry, furious and aggressive tone"
                    elif emotion == "sad":
                        control_instruction = f"{base_desc}, speaking with a sad, tearful, and melancholic voice, showing deep sorrow"
                    elif emotion == "happy":
                        control_instruction = f"{base_desc}, speaking with a happy, joyful, and cheerful tone, full of laughter"
                    elif emotion == "excited":
                        control_instruction = f"{base_desc}, speaking with an excited, enthusiastic and energetic voice"
                    elif emotion == "whisper":
                        control_instruction = f"{base_desc}, speaking in a low whisper, very soft and quiet voice"
                    elif emotion == "scared":
                        control_instruction = f"{base_desc}, speaking with a scared, trembling, and terrified voice, full of fear"
                    elif emotion == "surprised":
                        control_instruction = f"{base_desc}, speaking with a surprised, shocked, and amazed voice"
                    elif emotion == "calm":
                        control_instruction = f"{base_desc}, speaking with a calm, peaceful and relaxed tone"
                    elif emotion == "thought":
                        control_instruction = f"{base_desc}, speaking in a soft, low, introspective inner monologue tone"
                    else:
                        control_instruction = base_desc
                        
                    success = generate_local_voxcpm_tts(
                        clean_text, output_path, control_instruction, 
                        reference_wav=None, host=host, port=port
                    )
                
                if success:
                    return True
                
                if voice == "voxcpm-custom":
                    fallback_voice = "km-KH-SreymomNeural"
                    if custom_voice_path:
                        path_lower = custom_voice_path.lower()
                        if any(x in path_lower for x in ("male", "ប្រុស", "man", "boy", "guy")):
                            fallback_voice = "km-KH-PisethNeural"
                logger.warning(f"Local VoxCPM synthesis failed, falling back to {fallback_voice}...")
                voice = fallback_voice

            else:
                # backend == "openbmb"
                if voice == "voxcpm-custom":
                    control_instruction = ""
                    success = generate_voxcpm_tts(clean_text, output_path, control_instruction, reference_wav=custom_voice_path)
                else:
                    if "female" in voice:
                        base_desc = "A young woman with a sweet, clear and gentle voice"
                        fallback_voice = "km-KH-SreymomNeural"
                    elif "male" in voice:
                        base_desc = "A mature man with a clear, calm and professional voice"
                        fallback_voice = "km-KH-PisethNeural"
                    else:
                        base_desc = "A speaker with a clear and natural voice"
                        fallback_voice = "km-KH-SreymomNeural"
                        
                    if not emotion:
                        if "-angry" in voice: emotion = "angry"
                        elif "-sad" in voice: emotion = "sad"
                        elif "-happy" in voice: emotion = "happy"
                        elif "-excited" in voice: emotion = "excited"
                        elif "-whisper" in voice: emotion = "whisper"
                        elif "-scared" in voice: emotion = "scared"
                        elif "-surprised" in voice: emotion = "surprised"
                        elif "-calm" in voice: emotion = "calm"
                        elif "-thought" in voice: emotion = "thought"

                    if voice in ("voxcpm-female", "voxcpm-male"):
                        emotion = ""

                    if emotion == "angry":
                        control_instruction = f"{base_desc}, speaking with an angry, furious and aggressive tone"
                    elif emotion == "sad":
                        control_instruction = f"{base_desc}, speaking with a sad, tearful, and melancholic voice, showing deep sorrow"
                    elif emotion == "happy":
                        control_instruction = f"{base_desc}, speaking with a happy, joyful, and cheerful tone, full of laughter"
                    elif emotion == "excited":
                        control_instruction = f"{base_desc}, speaking with an excited, enthusiastic and energetic voice"
                    elif emotion == "whisper":
                        control_instruction = f"{base_desc}, speaking in a low whisper, very soft and quiet voice"
                    elif emotion == "scared":
                        control_instruction = f"{base_desc}, speaking with a scared, trembling, and terrified voice, full of fear"
                    elif emotion == "surprised":
                        control_instruction = f"{base_desc}, speaking with a surprised, shocked, and amazed voice"
                    elif emotion == "calm":
                        control_instruction = f"{base_desc}, speaking with a calm, peaceful and relaxed tone"
                    elif emotion == "thought":
                        control_instruction = f"{base_desc}, speaking in a soft, low, introspective inner monologue tone"
                    else:
                        control_instruction = base_desc
                        
                    success = generate_voxcpm_tts(clean_text, output_path, control_instruction)
                
                if success:
                    return True
                
                if voice == "voxcpm-custom":
                    fallback_voice = "km-KH-SreymomNeural"
                    if custom_voice_path:
                        path_lower = custom_voice_path.lower()
                        if any(x in path_lower for x in ("male", "ប្រុស", "man", "boy", "guy")):
                            fallback_voice = "km-KH-PisethNeural"
                logger.warning(f"Remote OpenBMB VoxCPM synthesis failed, falling back to {fallback_voice}...")
                voice = fallback_voice
        except Exception as e:
            logger.error(f"VoxCPM synthesis failed: {e}")
            voice = "km-KH-SreymomNeural"
        
    # Try using edge-tts if voice is provided and is not google-tts
    if voice and voice != "google-tts":
        try:
            import edge_tts
            import asyncio
            async def _generate():
                communicate = edge_tts.Communicate(clean_text, voice)
                await communicate.save(output_path)
            asyncio.run(_generate())
            return True
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed for voice '{voice}', falling back to Google: {e}")
            
    # Fallback to Google Translate TTS
    try:
        chunks = []
        words = clean_text.split(" ")
        current_chunk = ""
        for w in words:
            if len(current_chunk) + len(w) + 1 < 180:
                current_chunk += (" " if current_chunk else "") + w
            else:
                chunks.append(current_chunk)
                current_chunk = w
        if current_chunk:
            chunks.append(current_chunk)
        
        with open(output_path, "wb") as outfile:
            for idx, chunk in enumerate(chunks):
                url = "https://translate.google.com/translate_tts"
                params = {
                    "ie": "UTF-8",
                    "tl": lang,
                    "client": "tw-ob",
                    "q": chunk
                }
                query_string = urllib.parse.urlencode(params)
                req = urllib.request.Request(f"{url}?{query_string}", headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'
                })
                with urllib.request.urlopen(req, timeout=10) as response:
                    outfile.write(response.read())
        return True
    except Exception as e:
        logger.error(f"Google TTS generation failed for text '{clean_text}': {e}")
        return False

def get_audio_duration(file_path: str) -> float:
    """Gets audio file duration using ffprobe."""
    try:
        ffprobe = get_ffprobe_cmd()
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        cmd = [
            ffprobe, "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        output = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace", startupinfo=startupinfo, stderr=subprocess.DEVNULL).strip()
        return float(output)
    except Exception:
        return 0.0

# ==========================================
# AUDIO TIMELINE MIXING HELPERS (ABSOLUTE ALIGNMENT)
# ==========================================

def mix_pcm_clips(clips_info: list, video_duration: float, sample_rate: int = 24000) -> bytes:
    import numpy as np
    import wave
    total_samples = int(video_duration * sample_rate) + 24000
    mixed_data = np.zeros(total_samples, dtype=np.int16)
    
    for clip in clips_info:
        path = clip["path"]
        start_time = clip["start"]
        
        if not os.path.exists(path):
            continue
        try:
            with wave.open(path, "rb") as wav_in:
                nframes = wav_in.getnframes()
                raw_bytes = wav_in.readframes(nframes)
            
            clip_samples = np.frombuffer(raw_bytes, dtype=np.int16)
            start_sample = int(start_time * sample_rate)
            if start_sample >= total_samples:
                continue
                
            end_sample = start_sample + len(clip_samples)
            if end_sample > total_samples:
                clip_samples = clip_samples[:total_samples - start_sample]
                end_sample = total_samples
                
            temp = mixed_data[start_sample:end_sample].astype(np.int32) + clip_samples.astype(np.int32)
            mixed_data[start_sample:end_sample] = np.clip(temp, -32768, 32767).astype(np.int16)
        except Exception as e:
            logger.error(f"Failed to mix PCM clip {path}: {e}")
            
    return mixed_data.tobytes()

def trim_wav_silence(input_path: str, output_path: str, threshold: int = 150) -> bool:
    import numpy as np
    import wave
    try:
        with wave.open(input_path, "rb") as wav_in:
            params = wav_in.getparams()
            nframes = wav_in.getnframes()
            raw_bytes = wav_in.readframes(nframes)
        
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        abs_samples = np.abs(samples)
        non_silent_indices = np.where(abs_samples > threshold)[0]
        
        if len(non_silent_indices) > 0:
            first_idx = non_silent_indices[0]
            first_idx = max(0, first_idx - 240) # 10ms safety padding
            
            last_idx = non_silent_indices[-1]
            last_idx = min(len(samples) - 1, last_idx + 1200) # 50ms safety padding
            
            trimmed_samples = samples[first_idx:last_idx+1]
        else:
            trimmed_samples = samples
            
        trimmed_bytes = trimmed_samples.tobytes()
        
        with wave.open(output_path, "wb") as wav_out:
            wav_out.setparams(params)
            wav_out.writeframes(trimmed_bytes)
        return True
    except Exception as e:
        logger.error(f"Failed to trim silence from WAV {input_path}: {e}")
        return False

def detect_echo(samples, sample_rate=24000):
    """
    Detects if there is a significant echo or reverberation in the audio samples.
    Returns (has_echo, delay_ms, decay).
    """
    import numpy as np
    if samples is None or len(samples) < sample_rate * 0.5:
        return False, 0.0, 0.0
        
    try:
        # Take absolute values of samples
        abs_samples = np.abs(samples).astype(np.float32)
        
        # Smooth the envelope using moving average (window size of 5ms = 120 samples)
        window_size = int(sample_rate * 0.005)
        if window_size < 1:
            window_size = 1
        envelope = np.convolve(abs_samples, np.ones(window_size)/window_size, mode='same')
        
        # Subsample envelope to speed up autocorrelation (downsample by 40x)
        ds_factor = 40
        env_ds = envelope[::ds_factor]
        ds_sr = sample_rate / ds_factor
        
        # Compute autocorrelation
        n = len(env_ds)
        if n < 30:
            return False, 0.0, 0.0
            
        # Mean center
        env_ds = env_ds - np.mean(env_ds)
        
        # We check delays from 50ms to 250ms
        min_lag = int(ds_sr * 0.05) # 50ms
        max_lag = int(ds_sr * 0.25) # 250ms
        
        if max_lag >= n:
            max_lag = n - 1
        if min_lag >= max_lag:
            return False, 0.0, 0.0
            
        var = np.sum(env_ds ** 2)
        if var < 1e-6:
            return False, 0.0, 0.0
            
        best_lag = 0
        max_corr = 0.0
        
        for lag in range(min_lag, max_lag + 1):
            corr = np.sum(env_ds[lag:] * env_ds[:-lag]) / var
            if corr > max_corr:
                max_corr = corr
                best_lag = lag
                
        # If correlation peak is above threshold, we consider it has echo
        if max_corr > 0.22:
            delay_ms = (best_lag * ds_factor / sample_rate) * 1000.0
            delay_ms = max(50.0, min(300.0, delay_ms))
            decay = max(0.1, min(0.5, max_corr * 1.2 - 0.1))
            return True, delay_ms, decay
    except Exception:
        pass
        
    return False, 0.0, 0.0

def detect_pitch(samples, sample_rate=24000):
    """
    Estimates the fundamental frequency (F0) of an audio segment using autocorrelation,
    enhanced with Hanning windowing, Sondhi center-clipping, lag-bias normalization,
    and RMS energy weighting to drastically reduce octave errors.
    """
    import numpy as np
    if samples is None or len(samples) < int(sample_rate * 0.1): # less than 100ms
        return None
    
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    if rms < 50.0: # threshold for silence
        return None

    # Frame settings: 50ms frames, 25ms overlap
    frame_size = int(sample_rate * 0.05)
    hop_size = int(sample_rate * 0.025)
    
    # Human speech fundamental frequency limits (60 Hz to 360 Hz)
    min_lag = int(sample_rate / 360)
    max_lag = int(sample_rate / 60)
    
    pitches = []
    
    # Hanning window to prevent spectral leakage
    hanning_window = np.hanning(frame_size)
    
    # Pre-calculate autocorrelation of the Hanning window to normalize lag-bias
    win_corr = np.correlate(hanning_window, hanning_window, mode='full')[frame_size - 1:]
    
    for start_idx in range(0, len(samples) - frame_size, hop_size):
        frame = samples[start_idx : start_idx + frame_size].astype(np.float32)
        frame_rms = np.sqrt(np.mean(frame ** 2))
        
        # Skip silence/unvoiced/low-energy frames
        if frame_rms < 120.0:
            continue
            
        # Center-clipping at 30% of max amplitude to remove formant/harmonics interference
        max_val = np.max(np.abs(frame))
        if max_val > 0:
            clip_level = 0.3 * max_val
            # True Sondhi center-clipping
            clipped = np.zeros_like(frame)
            clipped[frame > clip_level] = frame[frame > clip_level] - clip_level
            clipped[frame < -clip_level] = frame[frame < -clip_level] + clip_level
            frame = clipped
            
        # DC removal
        frame = frame - np.mean(frame)
        
        # Apply windowing
        frame = frame * hanning_window
        
        # Autocorrelation
        n = len(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[n - 1 :]
        
        if len(corr) <= max_lag:
            continue
            
        # Normalize lag-bias using window autocorrelation
        corr_normalized = corr / (win_corr + 1e-6)
        
        search_area = corr_normalized[min_lag : max_lag + 1]
        if len(search_area) == 0:
            continue
            
        # Find all local maxima (peaks) in the search area
        peaks = []
        for i in range(1, len(search_area) - 1):
            if search_area[i] > search_area[i - 1] and search_area[i] > search_area[i + 1]:
                peaks.append((i + min_lag, search_area[i]))

        if len(peaks) > 0:
            # Find the global maximum correlation among the local peaks
            global_max_val = max(val for _, val in peaks)
            # Find the first peak (smallest lag / highest frequency) that is >= 80% of the global max
            threshold = 0.80 * global_max_val
            chosen_peak = None
            for lag, val in peaks:
                if val >= threshold:
                    chosen_peak = (lag, val)
                    break
            
            if chosen_peak:
                peak_idx, peak_val = chosen_peak
            else:
                peak_idx = np.argmax(search_area) + min_lag
                peak_val = search_area[peak_idx - min_lag]
        else:
            # Fallback to simple global max if no local peaks exist
            peak_idx = np.argmax(search_area) + min_lag
            peak_val = search_area[peak_idx - min_lag]
        
        zero_lag_val = corr_normalized[0]
        if zero_lag_val > 1e-6:
            norm_peak = peak_val / zero_lag_val
            
            # Voicing threshold (high correlation means periodic pitch)
            if norm_peak > 0.35:
                f0 = sample_rate / peak_idx
                if 60.0 <= f0 <= 360.0:
                    pitches.append((f0, frame_rms))
                    
    if len(pitches) == 0:
        return None
        
    # Compute energy-weighted average of pitch estimates
    total_weight = sum(w for _, w in pitches)
    if total_weight > 0:
        weighted_f0 = sum(f0 * w for f0, w in pitches) / total_weight
        return weighted_f0
    else:
        return np.median([f0 for f0, _ in pitches])

# Mapping of languages to their respective (Male, Female) voice neural configurations
GENDER_VOICES = {
    "km": {"male": "km-KH-PisethNeural", "female": "km-KH-SreymomNeural"},
    "en": {"male": "en-US-GuyNeural", "female": "en-US-AriaNeural"},
    "th": {"male": "th-TH-NiwatNeural", "female": "th-TH-AcharaNeural"},
    "vi": {"male": "vi-VN-NamMinhNeural", "female": "vi-VN-HoaiMyNeural"},
    "zh-cn": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
    "ja": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
    "ko": {"male": "ko-KR-InJoonNeural", "female": "ko-KR-SunHiNeural"},
    "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"}
}

def check_stereo_difference(video_path, ffmpeg_path, temp_dir):
    """
    Extracts a 10-second stereo WAV sample from the video and checks if L and R channels are different.
    Returns True if the video has true stereo (different channels), and False if it is mono or dual-mono.
    """
    import subprocess
    import wave
    import numpy as np
    
    test_wav = os.path.join(temp_dir, "stereo_test.wav")
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        
    # Extract first 10 seconds (or less) as stereo WAV
    cmd = [
        ffmpeg_path, "-y", "-i", video_path,
        "-t", "10", "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2",
        test_wav
    ]
    try:
        proc = subprocess.Popen(cmd, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait()
        
        if not os.path.exists(test_wav):
            return False
            
        with wave.open(test_wav, "rb") as wav_in:
            nchannels = wav_in.getnchannels()
            if nchannels < 2:
                return False
            nframes = wav_in.getnframes()
            raw_bytes = wav_in.readframes(nframes)
            
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        # Reshape to (N, 2)
        samples = samples.reshape(-1, 2)
        left = samples[:, 0].astype(np.float32)
        right = samples[:, 1].astype(np.float32)
        
        # Calculate mean absolute difference
        diff = np.mean(np.abs(left - right))
        mean_amp = np.mean(np.abs(left) + np.abs(right)) / 2.0
        
        if mean_amp < 10.0: # Very quiet/silent
            return False
            
        rel_diff = diff / mean_amp
        # If relative difference is >= 1%, we consider it true stereo
        return rel_diff >= 0.01
    except Exception:
        return False

class GeminiKeyVerifier(QThread):
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        
    def run(self):
        import json
        import urllib.request
        import urllib.error
        import re
        
        if not self.api_key:
            self.finished_signal.emit(False, "API Key is empty.")
            return
            
        parts = re.split(r'[\s,;\n]+', self.api_key)
        keys = [p.strip() for p in parts if p.strip()]
        
        if not keys:
            self.finished_signal.emit(False, "No valid keys found.")
            return
            
        valid_count = 0
        invalid_details = []
        
        for idx, key in enumerate(keys):
            masked_key = key[:7] + "..." + key[-4:] if len(key) > 11 else "Invalid Key Length"
            model_name = "gemini-2.5-flash"
            try:
                from settings.manager import load_settings
                model_name = load_settings().get("gemini_model", "gemini-2.5-flash")
            except Exception:
                pass
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Hello"}
                        ]
                    }
                ]
            }
            
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    if "candidates" in res_data:
                        valid_count += 1
                    else:
                        invalid_details.append(f"{masked_key}: Unexpected response format")
            except urllib.error.HTTPError as e:
                try:
                    err_data = json.loads(e.read().decode("utf-8"))
                    err_msg = err_data.get("error", {}).get("message", str(e))
                    invalid_details.append(f"{masked_key}: {err_msg}")
                except Exception:
                    invalid_details.append(f"{masked_key}: HTTP Error {e.code} {e.reason}")
            except Exception as e:
                invalid_details.append(f"{masked_key}: {str(e)}")
                
        if valid_count == len(keys):
            self.finished_signal.emit(True, f"🎉 All {len(keys)} Gemini API Keys are valid and working!")
        elif valid_count > 0:
            details_str = ", ".join(invalid_details)
            self.finished_signal.emit(True, f"⚠️ {valid_count} of {len(keys)} keys are valid. Errors: {details_str}")
        else:
            details_str = "; ".join(invalid_details)
            self.finished_signal.emit(False, f"❌ All keys failed verification: {details_str}")

class WhisperDownloadWorker(QThread):
    progress = pyqtSignal(str)
    progress_val = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, model_size, output_dir):
        super().__init__()
        self.model_size = model_size
        self.output_dir = output_dir
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            import urllib.request
            import time
            
            required_files = ['config.json', 'vocabulary.txt', 'tokenizer.json', 'model.bin']
            base_url = f"https://huggingface.co/Systran/faster-whisper-{self.model_size}/resolve/main"
            
            self.progress.emit(f"🔍 Analyzing Whisper '{self.model_size}' repository files...")
            
            file_sizes = {}
            total_size = 0
            for fname in required_files:
                if self.is_cancelled:
                    self.finished.emit(False, "Cancelled by user.")
                    return
                url = f"{base_url}/{fname}"
                try:
                    req = urllib.request.Request(url, method='HEAD')
                    with urllib.request.urlopen(req) as resp:
                        size = int(resp.headers.get('Content-Length', 0))
                        file_sizes[fname] = size
                        total_size += size
                except Exception as head_err:
                    self.progress.emit(f"⚠️ Failed to query size of {fname}: {head_err}")
                    file_sizes[fname] = 0
            
            self.progress.emit(f"📊 Total download size: {total_size / (1024 * 1024):.2f} MB")
            
            os.makedirs(self.output_dir, exist_ok=True)
            downloaded_bytes = 0
            
            start_time = time.time()
            for fname in required_files:
                if self.is_cancelled:
                    self.finished.emit(False, "Cancelled by user.")
                    return
                url = f"{base_url}/{fname}"
                dest_path = os.path.join(self.output_dir, fname)
                temp_dest_path = dest_path + ".tmp"
                
                self.progress.emit(f"📥 Downloading {fname}...")
                
                try:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req) as resp, open(temp_dest_path, "wb") as f_out:
                        chunk_size = 128 * 1024 # 128 KB
                        while True:
                            if self.is_cancelled:
                                f_out.close()
                                if os.path.exists(temp_dest_path):
                                    try:
                                        os.remove(temp_dest_path)
                                    except Exception:
                                        pass
                                self.finished.emit(False, "Cancelled by user.")
                                return
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            downloaded_bytes += len(chunk)
                            
                            if total_size > 0:
                                percent = int((downloaded_bytes / total_size) * 100)
                                self.progress_val.emit(percent)
                    
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(temp_dest_path, dest_path)
                except Exception as dl_err:
                    if os.path.exists(temp_dest_path):
                        try:
                            os.remove(temp_dest_path)
                        except Exception:
                            pass
                    raise dl_err
            
            duration = time.time() - start_time
            self.progress.emit(f"✨ Downloaded {total_size / (1024 * 1024):.2f} MB in {duration:.1f}s ({total_size / (1024 * 1024) / max(0.1, duration):.2f} MB/s)")
            self.finished.emit(True, f"🎉 Whisper '{self.model_size}' downloaded successfully to: {self.output_dir}")
        except Exception as e:
            self.finished.emit(False, str(e))

# ==========================================
# BACKGROUND WORKER FOR DUBBING PROCESS
# ==========================================

class DubberWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str) # success, output_path, details
    transcription_ready = pyqtSignal(str, list, list, dict) # video_path, segments, translations, params

    def __init__(self, params):
        super().__init__()
        self.video_path = os.path.abspath(params["video_path"])
        self.src_lang = params["src_lang"]
        self.tgt_lang = params["tgt_lang"]
        self.voice = params.get("voice")
        self.custom_voice_path = params.get("custom_voice_path")
        self.model_size = params["model_size"]
        self.device = params["device"]
        self.vol_original = params["vol_original"]
        self.vol_dubbed = params["vol_dubbed"]
        self.auto_speed = params["auto_speed"]
        self.mute_vocals = params.get("mute_vocals", False)
        self.mute_thoughts = params.get("mute_thoughts", False)
        self.match_echo = params.get("match_echo", False)
        self.telephone_effect = params.get("telephone_effect", False)
        self.output_dir = os.path.abspath(params["output_dir"])
        self.interactive = params.get("interactive", False)
        self.is_cancelled = False
        self.process = None
        self.resume_state = None

    def run(self):
        # Check global dub cache first to skip processing if already done
        if self.interactive:
            try:
                cache = load_dub_cache()
                if self.video_path in cache:
                    cached = cache[self.video_path]
                    segments = cached.get("segments")
                    translations = cached.get("translations")
                    cached_params = cached.get("params", {})
                    for key in ["vol_original", "vol_dubbed", "auto_speed", "mute_vocals", "match_echo", "telephone_effect"]:
                        if key in cached_params:
                            setattr(self, key, cached_params[key])
                    params_to_emit = {
                        "video_path": self.video_path,
                        "src_lang": self.src_lang,
                        "tgt_lang": self.tgt_lang,
                        "voice": self.voice,
                        "custom_voice_path": self.custom_voice_path,
                        "model_size": self.model_size,
                        "device": self.device,
                        "vol_original": self.vol_original,
                        "vol_dubbed": self.vol_dubbed,
                        "auto_speed": self.auto_speed,
                        "mute_vocals": self.mute_vocals,
                        "mute_thoughts": self.mute_thoughts,
                        "match_echo": self.match_echo,
                        "telephone_effect": self.telephone_effect,
                        "output_dir": self.output_dir,
                        "interactive": True,
                        "detected_voices": cached.get("voices", [])
                    }
                    self.log.emit("⚡ Found cached dubbing project segments and translations. Skipping execution!")
                    self.progress.emit(100)
                    self.status.emit("Completed")
                    self.transcription_ready.emit(self.video_path, segments, translations, params_to_emit)
                    self.finished.emit(True, "", "Loaded from cache")
                    return
            except Exception as ce:
                logger.error(f"Failed to load cached project: {ce}")

        temp_files = []
        try:
            self.status.emit("Initializing project directories...")
            self.progress.emit(5)
            self.log.emit(f"🎬 Source Video: {self.video_path}")
            
            # If resuming, load cached data if available
            resume_data = None
            if hasattr(self, "resume_state") and self.resume_state:
                resume_data = self.resume_state
                temp_dir = resume_data["temp_dir"]
                self.log.emit(f"🔄 Resuming job. Restoring working directory: {temp_dir}")
                os.makedirs(temp_dir, exist_ok=True)
            else:
                # Create a unique temp folder for processing clips
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="nova_dub_")
                self.log.emit(f"📁 Created working directory: {temp_dir}")
                
            # Initialize / save initial resume state
            if not resume_data:
                resume_data = {
                    "video_path": self.video_path,
                    "params": {
                        "video_path": self.video_path,
                        "src_lang": self.src_lang,
                        "tgt_lang": self.tgt_lang,
                        "voice": self.voice,
                        "custom_voice_path": self.custom_voice_path,
                        "model_size": self.model_size,
                        "device": self.device,
                        "vol_original": self.vol_original,
                        "vol_dubbed": self.vol_dubbed,
                        "auto_speed": self.auto_speed,
                        "mute_vocals": self.mute_vocals,
                        "mute_thoughts": self.mute_thoughts,
                        "match_echo": self.match_echo,
                        "telephone_effect": self.telephone_effect,
                        "output_dir": self.output_dir,
                        "interactive": self.interactive
                    },
                    "temp_dir": temp_dir,
                    "step": "initialized",
                    "segments": [],
                    "translations": []
                }
                save_resume_state(resume_data)
            
            ffmpeg_path = get_ffmpeg_cmd()
            
            # Step 1: Extract audio track from original video
            orig_audio_path = os.path.join(temp_dir, "original_audio.wav")
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
            has_audio = has_audio_stream(self.video_path)
            
            if resume_data and os.path.exists(orig_audio_path) and os.path.getsize(orig_audio_path) > 0:
                self.log.emit("🔄 Reusing extracted original audio track.")
            else:
                self.status.emit("Extracting original audio track...")
                self.progress.emit(10)
                if has_audio:
                    self.log.emit("🔊 Video contains audio. Extracting...")
                    cmd_extract = [
                        ffmpeg_path, "-y", "-i", self.video_path,
                        "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
                        orig_audio_path
                    ]
                    self.process = subprocess.Popen(cmd_extract, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.process.wait()
                    if self.is_cancelled:
                        self.cleanup(temp_dir)
                        clear_resume_state()
                        return
                else:
                    self.log.emit("🔇 Video does not contain audio. Generating empty base track...")
                    # We still create a silent track
                    cmd_extract = [
                        ffmpeg_path, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                        "-t", "10", "-acodec", "pcm_s16le", orig_audio_path
                    ]
                    self.process = subprocess.Popen(cmd_extract, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.process.wait()
                
            # Apply vocal isolation filter for cleaner transcription (suppressing background music)
            transcribe_audio_path = orig_audio_path
            if has_audio:
                try:
                    transcribe_audio_path = os.path.join(temp_dir, "transcribe_audio.wav")
                    if resume_data and os.path.exists(transcribe_audio_path) and os.path.getsize(transcribe_audio_path) > 0:
                        self.log.emit("🔄 Reusing pre-processed audio track.")
                    else:
                        self.log.emit("🔍 Pre-processing audio track (applying bandpass filter to suppress background music)...")
                        cmd_preprocess = [
                            ffmpeg_path, "-y", "-i", orig_audio_path,
                            "-af", "stereotools=slev=0.015625,afftdn=nr=28,highpass=f=80,lowpass=f=8500,dialoguenhance=original=0.1:enhance=2.8:voice=4.0,loudnorm",
                            "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
                            transcribe_audio_path
                        ]
                        self.process = subprocess.Popen(cmd_preprocess, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.process.wait()
                        if not os.path.exists(transcribe_audio_path):
                            transcribe_audio_path = orig_audio_path
                except Exception as preprocess_err:
                    self.log.emit(f"⚠️ Failed to apply audio filter: {preprocess_err}. Using original audio.")
                    transcribe_audio_path = orig_audio_path
                
            # Load original audio samples for echo and gender detection
            orig_all_samples = None
            if has_audio and (self.match_echo or self.voice in ("auto-gender", "voxcpm-auto", "voxcpm-auto-cloned")):
                try:
                    import wave
                    import numpy as np
                    with wave.open(orig_audio_path, "rb") as wav_in:
                        orig_nframes = wav_in.getnframes()
                        orig_bytes = wav_in.readframes(orig_nframes)
                        orig_all_samples = np.frombuffer(orig_bytes, dtype=np.int16)
                except Exception as e:
                    self.log.emit(f"⚠️ Failed to load original audio for echo/gender detection: {e}")

            # Step 2: Speech-To-Text Transcription (Whisper)
            segments_list = []
            if resume_data and resume_data.get("segments"):
                self.log.emit("🔄 Reusing cached Whisper transcription segments.")
                segments_list = resume_data["segments"]
                self.progress.emit(35)
            else:
                self.status.emit("Transcribing voice audio tracks...")
                self.progress.emit(20)
                self.log.emit("🤖 Loading Whisper Model...")
                
                # Attempt to import faster_whisper dynamically
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    self.log.emit("⚠️ faster-whisper package not installed. Running simulation mode.")
                    self.status.emit("faster-whisper missing! Simulating transcription...")
                    time.sleep(2)
                    
                    # Dummy segments
                    segments_list = [
                        {"start": 0.5, "end": 3.0, "text": "Hello, how are you? Welcome to Nova Ultimate Suite."},
                        {"start": 4.2, "end": 7.5, "text": "This video dubbing AI is generating automated Khmer voiceovers."}
                    ]
                else:
                    local_path = os.path.abspath(os.path.join(program_path, "models", "whisper", self.model_size))
                    is_local = is_whisper_model_valid(local_path, self.model_size)
                    
                    model_path_or_name = local_path if is_local else self.model_size
                    if is_local:
                        self.log.emit(f"💾 Found offline Whisper model locally. Loading from: {local_path}")
                    else:
                        self.log.emit("🌐 Loading Whisper model from Hugging Face cache (or downloading dynamically)...")
                        
                    if self.device == "cuda":
                        best_type = "float16"
                        try:
                            import ctranslate2
                            supported_types = ctranslate2.get_supported_compute_types("cuda")
                            if "float16" in supported_types:
                                best_type = "float16"
                            elif "int8" in supported_types:
                                best_type = "int8"
                            elif "float32" in supported_types:
                                best_type = "float32"
                        except Exception:
                            pass

                        # Detect selected/best GPU device index
                        gpu_id = 0
                        try:
                            from services.gpu_manager import GPUManager
                            gpu_id = GPUManager().get_best_gpu_device_id()
                        except Exception:
                            pass

                        try:
                            self.log.emit(f"⚙️ Loading Whisper '{self.model_size}' model on GPU ID {gpu_id} ({best_type})...")
                            model, is_cached = get_cached_whisper_model(model_path_or_name, "cuda", best_type, is_local, device_index=gpu_id)
                            if is_cached:
                                self.log.emit("⚡ Reusing cached Whisper model.")
                        except Exception as err:
                            self.log.emit(f"⚠️ GPU loading with {best_type} failed on Device ID {gpu_id}: {err}. Falling back to CPU...")
                            model, is_cached = get_cached_whisper_model(model_path_or_name, "cpu", "int8", is_local)
                            if is_cached:
                                self.log.emit("⚡ Reusing cached Whisper model.")
                    else:
                        self.log.emit(f"⚙️ Loading Whisper '{self.model_size}' model on CPU (int8)...")
                        model, is_cached = get_cached_whisper_model(model_path_or_name, "cpu", "int8", is_local)
                        if is_cached:
                            self.log.emit("⚡ Reusing cached Whisper model.")
                    
                    if self.is_cancelled:
                        self.cleanup(temp_dir)
                        clear_resume_state()
                        return
                    
                    self.log.emit("🔍 Scanning speech tracks (with VAD noise filter)...")
                    lang_param = None if self.src_lang == "Auto Detect" else self.src_lang
                    segments, info = model.transcribe(
                        transcribe_audio_path,
                        beam_size=5,
                        temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
                        condition_on_previous_text=False,
                        language=lang_param,
                        vad_filter=True,
                        vad_parameters=dict(min_speech_duration_ms=100, speech_pad_ms=500, threshold=0.20, min_silence_duration_ms=250),
                        word_timestamps=True,
                        no_speech_threshold=0.90
                    )
                    
                    self.log.emit(f"📈 Detected language: '{info.language}' with probability {info.language_probability:.2f}")
                    
                    def clean_repetitions(text: str) -> str:
                        if not text:
                            return text
                        
                        text = text.strip()
                        
                        # 1. Reduce excessive character repetition (e.g. 呵呵呵呵 -> 呵呵)
                        cleaned = []
                        i = 0
                        while i < len(text):
                            char = text[i]
                            count = 1
                            while i + count < len(text) and text[i + count] == char:
                                count += 1
                            if count >= 4:
                                cleaned.append(char * 2)
                            else:
                                cleaned.append(text[i:i+count])
                            i += count
                        text = "".join(cleaned)
                        
                        # 2. Reduce excessive phrase repetition (e.g. 谢谢大家谢谢大家谢谢大家 -> 谢谢大家谢谢大家)
                        for length in range(2, 12):
                            for start in range(len(text) - length * 3):
                                pattern = text[start:start + length]
                                if text[start + length:start + length * 2] == pattern and text[start + length * 2:start + length * 3] == pattern:
                                    rep_count = 3
                                    while start + length * (rep_count + 1) <= len(text) and text[start + length * rep_count : start + length * (rep_count + 1)] == pattern:
                                        rep_count += 1
                                    full_rep = pattern * rep_count
                                    text = text.replace(full_rep, pattern * 2, 1)
                        return text.strip()

                    segments_list = []
                    for segment in segments:
                        if self.is_cancelled:
                            self.cleanup(temp_dir)
                            clear_resume_state()
                            return
                        # Ignore segments that are highly likely to be music or background noise
                        if getattr(segment, 'no_speech_prob', 0.0) > 0.98:
                            self.log.emit(f"🔇 Skipped noise segment: [{segment.start:.1f}s - {segment.end:.1f}s] (prob: {segment.no_speech_prob:.2f})")
                            continue
                        
                        # Clean up any repetition loops in the transcribed text
                        cleaned_text = clean_repetitions(segment.text)
                        
                        # Skip segments that become empty or contain only meaningless repeating characters
                        if not cleaned_text or cleaned_text in [".", ",", "!", "?", "，", "。", "！", "？"]:
                            continue
                        
                        # Extract exact word boundaries for millisecond-level precision
                        seg_start = segment.start
                        seg_end = segment.end
                        if hasattr(segment, 'words') and segment.words:
                            words = list(segment.words)
                            if words:
                                seg_start = words[0].start
                                seg_end = words[-1].end
                                
                        segments_list.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": cleaned_text
                        })
                        self.log.emit(f"📝 Subtitle: [{seg_start:.2f}s - {seg_end:.2f}s] {cleaned_text}")

                    # Keep the model in global cache for subsequent tasks
                    pass
                
                if resume_data:
                    resume_data["segments"] = segments_list
                    resume_data["step"] = "transcribed"
                    save_resume_state(resume_data)

            if not segments_list:
                self.log.emit("⚠️ No speech tracks detected in video file.")
                self.finished.emit(False, "", "No speech detected.")
                self.cleanup(temp_dir)
                return

            # Step 3 & 4: Translation and TTS synthesis
            translated_texts = []
            if resume_data and resume_data.get("translations"):
                self.log.emit("🔄 Reusing cached translations.")
                translated_texts = resume_data["translations"]
                self.progress.emit(45)
            else:
                self.status.emit("Translating text in batch...")
                self.progress.emit(45)
                try:
                    from settings.manager import load_settings
                    settings = load_settings()
                    gemini_key = settings.get("gemini_api_key", "").strip()
                except Exception:
                    gemini_key = ""
                
                if gemini_key:
                    self.log.emit("🌐 Translating all speech segments in batch using Gemini API...")
                else:
                    self.log.emit("🌐 Translating all speech segments in batch using Google Translate...")
                    
                orig_texts = [seg["text"] for seg in segments_list]
                src_lang_param = "auto"
                if self.src_lang and self.src_lang != "Auto Detect":
                    src_lang_param = self.src_lang
                translated_texts = translate_texts_batch(orig_texts, target_lang=self.tgt_lang, source_lang=src_lang_param)
                
                if resume_data:
                    resume_data["translations"] = translated_texts
                    resume_data["step"] = "translated"
                    save_resume_state(resume_data)
            
            if getattr(self, "interactive", False):
                dominant_gender = "female"
                if self.voice in ("auto-gender", "voxcpm-auto", "voxcpm-auto-cloned") and orig_all_samples is not None and len(orig_all_samples) > 0:
                    self.log.emit("📊 Analyzing overall speaker gender of the video for interactive workstation...")
                    all_pitches = []
                    for seg in segments_list:
                        try:
                            start_sample = int(seg["start"] * 24000)
                            end_sample = int(seg["end"] * 24000)
                            segment_samples = orig_all_samples[start_sample:end_sample]
                            pitch = detect_pitch(segment_samples, sample_rate=24000)
                            if pitch:
                                all_pitches.append(pitch)
                        except Exception:
                            pass
                    if all_pitches:
                        import numpy as np
                        median_pitch = np.median(all_pitches)
                        if median_pitch < 175.0:
                            dominant_gender = "male"
                        else:
                            dominant_gender = "female"
                        self.log.emit(f"📊 Overall video dominant gender: {dominant_gender} (median pitch: {median_pitch:.1f}Hz)")

                detected_voices = []
                for idx, seg in enumerate(segments_list):
                    detected_gender = dominant_gender
                    
                    # Try to detect gender of this specific segment
                    if self.voice in ("auto-gender", "voxcpm-auto", "voxcpm-auto-cloned") and orig_all_samples is not None and len(orig_all_samples) > 0:
                        try:
                            start_sample = int(seg["start"] * 24000)
                            end_sample = int(seg["end"] * 24000)
                            segment_samples = orig_all_samples[start_sample:end_sample]
                            pitch = detect_pitch(segment_samples, sample_rate=24000)
                            if pitch:
                                if pitch < 175.0:
                                    detected_gender = "male"
                                else:
                                    detected_gender = "female"
                                logger.info(f"Interactive segment {idx+1} pitch: {pitch:.1f} Hz -> {detected_gender.upper()}")
                        except Exception as e:
                            logger.warning(f"Failed to detect segment {idx+1} pitch: {e}")
                            
                    if self.voice == "voxcpm-auto-cloned":
                        try:
                            from settings.manager import load_settings
                            settings = load_settings()
                            saved_voices = settings.get("custom_cloned_voices", [])
                            selected_clone_path = resolve_auto_cloned_voice(
                                detected_gender, saved_voices, self.custom_voice_path, ffmpeg_path, startupinfo
                            )
                        except Exception as e:
                            logger.warning(f"Failed to resolve auto cloned voice in interactive: {e}")
                            selected_clone_path = self.custom_voice_path
                        segment_voice = f"voxcpm-custom|{selected_clone_path}"
                    elif self.voice == "voxcpm-auto":
                        _, emotion = extract_emotion(translated_texts[idx])
                        if emotion:
                            segment_voice = f"voxcpm-{detected_gender}-{emotion}"
                        else:
                            segment_voice = f"voxcpm-{detected_gender}"
                    elif self.voice == "auto-gender":
                        lang_voices = GENDER_VOICES.get(self.tgt_lang, GENDER_VOICES["km"])
                        segment_voice = lang_voices.get(detected_gender, lang_voices["female"])
                    else:
                        segment_voice = self.voice
                    
                    detected_voices.append(segment_voice)

                job_params = {
                    "src_lang": self.src_lang,
                    "tgt_lang": self.tgt_lang,
                    "voice": self.voice,
                    "custom_voice_path": self.custom_voice_path,
                    "model_size": self.model_size,
                    "device": self.device,
                    "vol_original": self.vol_original,
                    "vol_dubbed": self.vol_dubbed,
                    "auto_speed": self.auto_speed,
                    "mute_vocals": self.mute_vocals,
                    "mute_thoughts": self.mute_thoughts,
                    "match_echo": self.match_echo,
                    "output_dir": self.output_dir,
                    "detected_voices": detected_voices
                }
                try:
                    cache = load_dub_cache()
                    cache[self.video_path] = {
                        "segments": segments_list,
                        "translations": translated_texts,
                        "voices": detected_voices,
                        "params": job_params
                    }
                    save_dub_cache(cache)
                except Exception as ce:
                    logger.error(f"Failed to cache new project: {ce}")
                    
                self.transcription_ready.emit(self.video_path, segments_list, translated_texts, job_params)
                self.finished.emit(True, "", "Interactive transcription completed.")
                return
            
            # Pre-pass: Analyze dominant speaker gender of the entire video
            dominant_gender = "female"  # fallback default
            if self.voice in ("auto-gender", "voxcpm-auto", "voxcpm-auto-cloned") and orig_all_samples is not None and len(orig_all_samples) > 0:
                self.log.emit("📊 Analyzing overall speaker gender of the video...")
                all_pitches = []
                for seg in segments_list:
                    try:
                        start_sample = int(seg["start"] * 24000)
                        end_sample = int(seg["end"] * 24000)
                        segment_samples = orig_all_samples[start_sample:end_sample]
                        pitch = detect_pitch(segment_samples, sample_rate=24000)
                        if pitch:
                            all_pitches.append(pitch)
                    except Exception:
                        pass
                if all_pitches:
                    import numpy as np
                    median_pitch = np.median(all_pitches)
                    if median_pitch < 175.0:
                        dominant_gender = "male"
                    else:
                        dominant_gender = "female"
                    self.log.emit(f"📊 Audio Analysis Result: Median Pitch = {median_pitch:.1f} Hz. Dominant gender resolved to: {dominant_gender.upper()}")
                else:
                    self.log.emit("📊 Audio Analysis: Pitch could not be detected on any segment. Defaulting fallback to FEMALE.")

            # --- Parallel TTS Task Preparation ---
            self.status.emit("Preparing TTS tasks...")
            self.progress.emit(40)
            
            from settings.manager import load_settings
            settings = load_settings()
            backend = settings.get("voxcpm_backend", "local")
            host = settings.get("voxcpm_host", "127.0.0.1")
            port = int(settings.get("voxcpm_port", 8000))
            
            tts_tasks = []
            any_voxcpm = False
            
            for idx, seg in enumerate(segments_list):
                original_text = seg["text"].strip()
                translated_text = translated_texts[idx]
                
                # Determine voice (using pitch detection)
                segment_voice = None
                if self.voice in ("auto-gender", "voxcpm-auto", "voxcpm-auto-cloned"):
                    detected_gender = dominant_gender
                    
                    if orig_all_samples is not None and len(orig_all_samples) > 0:
                        try:
                            start_sample = int(seg["start"] * 24000)
                            end_sample = int(seg["end"] * 24000)
                            segment_samples = orig_all_samples[start_sample:end_sample]
                            pitch = detect_pitch(segment_samples, sample_rate=24000)
                            if pitch:
                                if pitch < 175.0:
                                    detected_gender = "male"
                                else:
                                    detected_gender = "female"
                                self.log.emit(f"📊 Segment {idx+1} speaker gender pitch: {pitch:.1f} Hz -> {detected_gender.upper()}")
                        except Exception as e:
                            logger.warning(f"Failed to detect segment {idx+1} pitch: {e}")
                            
                    if self.voice == "voxcpm-auto-cloned":
                        try:
                            saved_voices = settings.get("custom_cloned_voices", [])
                            selected_clone_path = resolve_auto_cloned_voice(
                                detected_gender, saved_voices, self.custom_voice_path, ffmpeg_path, startupinfo
                            )
                        except Exception as e:
                            logger.warning(f"Failed to resolve auto cloned voice in batch: {e}")
                            selected_clone_path = self.custom_voice_path
                        segment_voice = f"voxcpm-custom|{selected_clone_path}"
                    elif self.voice == "voxcpm-auto":
                        _, emotion = extract_emotion(translated_text)
                        if emotion:
                            segment_voice = f"voxcpm-{detected_gender}-{emotion}"
                        else:
                            segment_voice = f"voxcpm-{detected_gender}"
                    else:
                        lang_voices = GENDER_VOICES.get(self.tgt_lang, GENDER_VOICES["km"])
                        segment_voice = lang_voices.get(detected_gender, lang_voices["female"])
                else:
                    segment_voice = self.voice
                
                if segment_voice and "voxcpm" in segment_voice:
                    any_voxcpm = True
                    
                raw_tts_path = os.path.join(temp_dir, f"raw_tts_{idx}.mp3")
                tts_tasks.append({
                    "idx": idx,
                    "seg": seg,
                    "text": translated_text,
                    "voice": segment_voice,
                    "path": raw_tts_path
                })

            # Auto-start local VoxCPM server if offline or model weights not loaded
            if any_voxcpm and backend == "local":
                try:
                    from app.voxcpm_manager import LocalVoxCPMServerManager
                    manager = LocalVoxCPMServerManager()
                    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "VoxCPM2")
                    model_dir = settings.get("voxcpm_model_dir", default_dir)
                    device = get_safe_device(settings.get("voxcpm_device", "cuda"))
                    manager.configure(host, port, device, model_dir)
                    
                    running, loaded, _ = manager.is_running()
                    if not running:
                        self.log.emit("🟢 Local VoxCPM server is offline. Starting server automatically...")
                        started, msg = manager.start_server()
                        if not started:
                            logger.error(f"Failed to auto-start local VoxCPM server: {msg}")
                        else:
                            self.log.emit("🟢 Local VoxCPM server started. Waiting for model weights to load...")
                            import time
                            start_wait = time.time()
                            while time.time() - start_wait < 150:
                                is_run, is_loaded, error_msg = manager.is_running()
                                if error_msg:
                                    logger.error(f"Local VoxCPM server failed to load model weights: {error_msg}")
                                    break
                                if is_run and is_loaded:
                                    self.log.emit("🟢 Local VoxCPM model loaded successfully.")
                                    break
                                time.sleep(2)
                            else:
                                logger.warning("Timed out waiting for local VoxCPM model weights to load.")
                    elif not loaded:
                        self.log.emit("🟢 Local VoxCPM server is active but model weights are still loading. Waiting...")
                        import time
                        start_wait = time.time()
                        while time.time() - start_wait < 150:
                            is_run, is_loaded, error_msg = manager.is_running()
                            if error_msg:
                                logger.error(f"Local VoxCPM model weights loading failed: {error_msg}")
                                break
                            if is_run and is_loaded:
                                self.log.emit("🟢 Local VoxCPM model loaded successfully.")
                                break
                            time.sleep(2)
                        else:
                            logger.warning("Timed out waiting for local VoxCPM model weights to load.")
                except Exception as auto_start_err:
                    logger.error(f"Error during local VoxCPM auto-start check: {auto_start_err}")

            # Concurrency limit based on backend
            if any_voxcpm:
                if backend == "local":
                    max_workers = 2 # safe limit for GPU VRAM to avoid OOM
                else:
                    max_workers = 4 # safe limit for HF Space queue
            else:
                max_workers = 5 # higher concurrency for Edge-TTS / Google-TTS

            self.log.emit(f"🚀 Starting parallel speech synthesis (max concurrency: {max_workers} tasks)...")
            completed_tasks = 0
            
            def run_tts_task(task):
                if self.is_cancelled:
                    return task["idx"], False
                # If resuming, check if raw tts file already exists
                if os.path.exists(task["path"]) and os.path.getsize(task["path"]) > 0:
                    return task["idx"], True
                try:
                    success = generate_tts_file(
                        task["text"], 
                        task["path"], 
                        lang=self.tgt_lang, 
                        voice=task["voice"], 
                        custom_voice_path=self.custom_voice_path
                    )
                    return task["idx"], success
                except Exception as e:
                    logger.error(f"Error in parallel TTS task {task['idx']}: {e}")
                    return task["idx"], False

            tts_results = {}
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_tts_task, t): t for t in tts_tasks}
                for fut in as_completed(futures):
                    idx, success = fut.result()
                    tts_results[idx] = success
                    completed_tasks += 1
                    self.status.emit(f"Synthesizing voice {completed_tasks}/{len(tts_tasks)}...")
                    self.progress.emit(int(45 + (completed_tasks / len(tts_tasks)) * 35))

            # --- Sequential Post-Processing ---
            tts_clips = []
            
            for idx, task in enumerate(tts_tasks):
                if self.is_cancelled:
                    self.cleanup(temp_dir)
                    clear_resume_state()
                    return
                
                seg = task["seg"]
                translated_text = task["text"]
                raw_tts_path = task["path"]
                
                # Check if adjusted wav already exists from previous run
                adjusted_tts_path = os.path.join(temp_dir, f"adj_tts_{idx}.wav")
                if os.path.exists(adjusted_tts_path) and os.path.getsize(adjusted_tts_path) > 0:
                    self.log.emit(f"🔄 Reusing post-processed clip for segment {idx+1}")
                    final_clip_dur = get_audio_duration(adjusted_tts_path)
                    tts_clips.append({
                        "path": adjusted_tts_path,
                        "start": seg["start"],
                        "end": seg["start"] + final_clip_dur,
                        "text": translated_text
                    })
                    continue

                success = tts_results.get(idx, False)
                
                self.log.emit(f"🌐 [{idx+1}] Post-processing audio: '{seg['text'].strip()}' ➔ '{translated_text}'")
                
                if not success or not os.path.exists(raw_tts_path):
                    self.log.emit(f"❌ Failed to generate TTS for segment {idx+1}")
                    continue
                
                # Convert MP3 to raw mono WAV
                raw_wav_path = os.path.join(temp_dir, f"raw_wav_{idx}.wav")
                cmd_conv = [
                    ffmpeg_path, "-y", "-i", raw_tts_path,
                    "-map_metadata", "-1",
                    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", raw_wav_path
                ]
                self.process = subprocess.Popen(cmd_conv, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.process.wait()
                
                # Trim silence from raw WAV
                trimmed_wav_path = os.path.join(temp_dir, f"trimmed_{idx}.wav")
                success_trim = trim_wav_silence(raw_wav_path, trimmed_wav_path, threshold=150)
                if not success_trim or not os.path.exists(trimmed_wav_path):
                    trimmed_wav_path = raw_wav_path # Fallback to untrimmed if error
                
                # Check actual trimmed speech duration
                tts_dur = get_audio_duration(trimmed_wav_path)
                orig_dur = seg["end"] - seg["start"]
                
                # Detect echo of original segment
                has_segment_echo = False
                echo_delay = 0.0
                echo_decay = 0.0
                if orig_all_samples is not None:
                    try:
                        start_sample = int(seg["start"] * 24000)
                        end_sample = int(seg["end"] * 24000)
                        segment_samples = orig_all_samples[start_sample:end_sample]
                        has_segment_echo, echo_delay, echo_decay = detect_echo(segment_samples, sample_rate=24000)
                        if has_segment_echo:
                            self.log.emit(f"🔊 Echo detected in original segment {idx+1}: delay={echo_delay:.0f}ms, decay={echo_decay:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to detect echo for segment {idx+1}: {e}")
                
                adjusted_tts_path = os.path.join(temp_dir, f"adj_tts_{idx}.wav")
                
                # Build audio filters (atempo, aecho, or both)
                filters = []
                speed = 1.0
                if self.auto_speed and orig_dur > 0.1:
                    speed = tts_dur / orig_dur
                    if speed > 1.45:
                        speed = 1.45
                    elif speed < 1.0:
                        speed = 1.0
                    
                    if abs(speed - 1.0) > 0.01:
                        filters.append(f"atempo={speed}")
                        if speed > 1.0:
                            self.log.emit(f"⚡ Segment {idx+1}: Speeding up by {speed:.2f}x to fit timeline.")
                        else:
                            self.log.emit(f"⚡ Segment {idx+1}: Slowing down by {speed:.2f}x to fit timeline.")
                            
                    
                if filters:
                    filter_str = ",".join(filters)
                    cmd_speed = [
                        ffmpeg_path, "-y", "-i", trimmed_wav_path,
                        "-filter:a", filter_str,
                        "-map_metadata", "-1",
                        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", adjusted_tts_path
                    ]
                    self.process = subprocess.Popen(cmd_speed, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.process.wait()
                else:
                    # Simply copy trimmed wav
                    import shutil
                    shutil.copyfile(trimmed_wav_path, adjusted_tts_path)
                    
                final_clip_dur = get_audio_duration(adjusted_tts_path)
                
                tts_clips.append({
                    "path": adjusted_tts_path,
                    "start": seg["start"],
                    "end": seg["start"] + final_clip_dur,
                    "text": translated_text
                })

            # Step 5: Construct dubbed voice track using numpy PCM mixing (eliminating timing drift and overlaps)
            self.status.emit("Mixing audio tracks to absolute timeline...")
            self.progress.emit(85)
            
            # Determine total duration of video
            video_duration = get_audio_duration(orig_audio_path)
            if video_duration <= 0.0:
                video_duration = 300.0 # Fallback 5 minutes
                
            self.log.emit(f"⏳ Video audio track duration: {video_duration:.2f} seconds")
            self.log.emit("🎛️ Commencing absolute audio slot alignment...")
            
            # Mix clips using numpy PCM mixer
            raw_pcm_bytes = mix_pcm_clips(tts_clips, video_duration, sample_rate=24000)
            
            # Write final dubbed speech wav file
            import wave
            dubbed_speech_wav = os.path.join(temp_dir, "dubbed_speech.wav")
            with wave.open(dubbed_speech_wav, "wb") as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2) # 16-bit
                wav_out.setframerate(24000)
                wav_out.writeframes(raw_pcm_bytes)
                
            self.log.emit("🔊 Timeline audio alignment complete (0 timing drift).")
            
            # Step 6: Render / Mix Dubbed Audio and Video
            self.status.emit("Mixing audio and rendering video...")
            self.progress.emit(90)
            
            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
            output_filename = f"{video_basename}_dubbed_{self.tgt_lang}.mp4"
            final_output_path = os.path.join(self.output_dir, output_filename)
            
            # Audio Mixing Filter complex (Ducking and mixing)
            # volume=vol_original for original sound track
            # volume=vol_dubbed for synthesized dubbed sound track
            if has_audio and self.vol_original > 0.001:
                channels = get_audio_channels(self.video_path)
                self.log.emit(f"🔊 Original video audio channels detected: {channels}")
                
                is_true_stereo = False
                if channels >= 2:
                    is_true_stereo = check_stereo_difference(self.video_path, ffmpeg_path, temp_dir)
                    self.log.emit(f"🔊 Video has true stereo background: {is_true_stereo}")
                
                # Identify all speech intervals to duck/mute the original audio during dubbing (preventing voice leakage/echo)
                duck_intervals = []
                for idx, seg in enumerate(segments_list):
                    start = max(0.0, seg["start"] - 0.1)
                    end = seg["end"] + 0.2
                    duck_intervals.append((start, end))
                
                if duck_intervals:
                    sum_between = "+".join([f"between(t,{s:.3f},{e:.3f})" for s, e in duck_intervals])
                    vol_filter = f"volume='if({sum_between},{self.vol_original * 0.05:.3f},{self.vol_original:.2f})':eval=frame"
                    self.log.emit(f"🔇 Professional Audio Ducking enabled: Reducing original background/echo volume to 5% during {len(duck_intervals)} speech segments.")
                else:
                    vol_filter = f"volume={self.vol_original:.2f}"
                
                if self.mute_vocals and is_true_stereo:
                    # Vocal Removal + BGM Keep:
                    # 1. Phase-Cancellation (pan=stereo|c0=c0-c1|c1=c0-c1) removes centre-panned vocals.
                    # 2. BGM is DUCKED (reduced to 15%) during dubbed speech segments so dubbed voice is clear.
                    # 3. BGM plays at full 100% volume between speech segments.
                    # Result: Vocals removed, BGM audible throughout, dubbed voice is clear.
                    if duck_intervals:
                        sum_speech = "+".join([f"between(t,{s:.3f},{e:.3f})" for s, e in duck_intervals])
                        bgm_vol_filter = f"volume='if({sum_speech},0.05,1.0)':eval=frame"
                        self.log.emit(f"🎵 Vocal Removal + BGM Keep: Phase-Cancellation applied. BGM ducked to 5% during {len(duck_intervals)} speech segments, 100% between segments. Dubbed vol: {self.vol_dubbed}...")
                    else:
                        bgm_vol_filter = "volume=1.0"
                        self.log.emit(f"🎵 Vocal Removal + BGM Keep: Phase-Cancellation applied. BGM at full volume. Dubbed vol: {self.vol_dubbed}...")
                    cmd_mix = [
                        ffmpeg_path, "-y", "-i", self.video_path, "-i", dubbed_speech_wav,
                        "-filter_complex", f"[0:a]pan=stereo|c0=c0-c1|c1=c0-c1,{bgm_vol_filter}[a0];[1:a]volume={self.vol_dubbed:.2f}[a1];[a0][a1]amix=inputs=2:duration=first[out_a]",
                        "-map", "0:v", "-map", "[out_a]", "-c:v", "copy", final_output_path
                    ]
                elif self.mute_vocals and not is_true_stereo:
                    # Mono audio — cannot phase-cancel vocals, but we still MIX the original audio
                    # (ducked to 5% during speech segments) with the dubbed track so background music is preserved.
                    self.log.emit("⚠️ Original video audio is Mono — vocal phase-cancellation not possible, but keeping BGM by mixing.")
                    self.log.emit(f"🎵 Mono BGM Keep: Mixing original audio (ducked to 1.5% during speech) + dubbed track (vol: {self.vol_dubbed})...")
                    if duck_intervals:
                        sum_between_mono = "+".join([f"between(t,{s:.3f},{e:.3f})" for s, e in duck_intervals])
                        mono_vol_filter = f"volume='if({sum_between_mono},0.015,{self.vol_original:.2f})':eval=frame"
                    else:
                        mono_vol_filter = f"volume={self.vol_original:.2f}"
                    cmd_mix = [
                        ffmpeg_path, "-y", "-i", self.video_path, "-i", dubbed_speech_wav,
                        "-filter_complex", f"[0:a]{mono_vol_filter}[a0];[1:a]volume={self.vol_dubbed:.2f}[a1];[a0][a1]amix=inputs=2:duration=first[out_a]",
                        "-map", "0:v", "-map", "[out_a]", "-c:v", "copy", final_output_path
                    ]
                else:
                    self.log.emit(f"🔊 Merging original audio (vol: {self.vol_original}) and dubbed tracks (vol: {self.vol_dubbed})...")
                    cmd_mix = [
                        ffmpeg_path, "-y", "-i", self.video_path, "-i", dubbed_speech_wav,
                        "-filter_complex", f"[0:a]{vol_filter}[a0];[1:a]volume={self.vol_dubbed:.2f}[a1];[a0][a1]amix=inputs=2:duration=first[out_a]",
                        "-map", "0:v", "-map", "[out_a]", "-c:v", "copy", final_output_path
                    ]
            else:
                self.log.emit("🔇 Playing pure dubbed voice track...")
                cmd_mix = [
                    ffmpeg_path, "-y", "-i", self.video_path, "-i", dubbed_speech_wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-shortest", final_output_path
                ]
                
            self.process = subprocess.Popen(cmd_mix, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process.wait()
            
            if self.is_cancelled:
                self.cleanup(temp_dir)
                return
                
            if os.path.exists(final_output_path):
                self.progress.emit(100)
                self.status.emit("Dubbing completed successfully!")
                self.log.emit(f"🎉 Dubbed Video saved to: {final_output_path}")
                self.finished.emit(True, final_output_path, f"Successfully dubbed to {self.tgt_lang.upper()}")
            else:
                self.finished.emit(False, "", "FFmpeg render error: Output file not generated.")
                
            # Cleanup temporary working files
            self.cleanup(temp_dir)
            
        except Exception as e:
            logger.exception(f"DubberWorker thread failed: {e}")
            self.log.emit(f"❌ Error: {str(e)}")
            self.finished.emit(False, "", str(e))
            self.cleanup(temp_dir)

    def cleanup(self, temp_dir):
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.log.emit("🧹 Cleaned up temporary cache directories.")
            clear_resume_state()
        except Exception as ce:
            logger.error(f"Failed to cleanup temp dir: {ce}")

    def cancel(self):
        self.is_cancelled = True
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass


# ==========================================
# USER INTERFACE PAGE
# ==========================================

VOICES_MAP = {
    "km": [
        ("Auto Piseth/Sreymom", "auto-gender"),
        ("Auto VoxCPM Male/Female", "voxcpm-auto"),
        ("Custom Voice Cloning", "voxcpm-custom"),
        ("VoxCPM ស្រីនាង (Female AI)", "voxcpm-female"),
        ("VoxCPM ពិសិដ្ឋ (Male AI)", "voxcpm-male"),
        ("Google TTS (Universal)", "google-tts"),
        ("Sreymom (Female)", "km-KH-SreymomNeural"),
        ("Piseth (Male)", "km-KH-PisethNeural")
    ],
    "en": [
        ("Auto Speaker (Male/Female)", "auto-gender"),
        ("Guy (Male)", "en-US-GuyNeural"),
        ("Aria (Female)", "en-US-AriaNeural")
    ],
    "th": [
        ("เสียงชาย/หญิงอัตโนมัติ (Niwat/Achara)", "auto-gender"),
        ("Niwat (Male)", "th-TH-NiwatNeural"),
        ("Achara (Female)", "th-TH-AcharaNeural")
    ],
    "vi": [
        ("Giọng Nam/Nữ Tự Động (Nam Minh/Hoài My)", "auto-gender"),
        ("Nam Minh (Male)", "vi-VN-NamMinhNeural"),
        ("Hoai My (Female)", "vi-VN-HoaiMyNeural")
    ],
    "zh-cn": [
        ("自动男女声切换 (云希/晓晓)", "auto-gender"),
        ("Yunxi (Male)", "zh-CN-YunxiNeural"),
        ("Xiaoxiao (Female)", "zh-CN-XiaoxiaoNeural")
    ],
    "ja": [
        ("自動男女音声切り替え (ケイタ/ナナミ)", "auto-gender"),
        ("Keita (Male)", "ja-JP-KeitaNeural"),
        ("Nanami (Female)", "ja-JP-NanamiNeural")
    ],
    "ko": [
        ("자동 남성/여성 목소리 (인준/선희)", "auto-gender"),
        ("InJoon (Male)", "ko-KR-InJoonNeural"),
        ("SunHi (Female)", "ko-KR-SunHiNeural")
    ],
    "fr": [
        ("Voix Homme/Femme Automatique (Henri/Denise)", "auto-gender"),
        ("Henri (Male)", "fr-FR-HenriNeural"),
        ("Denise (Female)", "fr-FR-DeniseNeural")
    ],
    "es": [
        ("Voz de Hombre/Mujer Automática (Álvaro/Elvira)", "auto-gender"),
        ("Alvaro (Male)", "es-ES-AlvaroNeural"),
        ("Elvira (Female)", "es-ES-ElviraNeural")
    ]
}

class ExtractOrigAudioWorker(QThread):
    finished = pyqtSignal(bool, str) # success, output_path or error
    
    def __init__(self, video_path, start_time, end_time, output_path):
        super().__init__()
        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.output_path = output_path
        
    def run(self):
        try:
            import subprocess
            ffmpeg_path = get_ffmpeg_cmd()
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            cmd = [
                ffmpeg_path, "-y",
                "-ss", f"{self.start_time:.3f}",
                "-to", f"{self.end_time:.3f}",
                "-i", self.video_path,
                "-vn",
                "-ac", "1", "-ar", "24000",
                self.output_path
            ]
            p = subprocess.Popen(cmd, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p.wait()
            if os.path.exists(self.output_path):
                self.finished.emit(True, self.output_path)
            else:
                self.finished.emit(False, "FFmpeg failed to extract original audio segment.")
        except Exception as e:
            self.finished.emit(False, str(e))


class SegmentTTSWorker(QThread):
    finished = pyqtSignal(bool, str) # success, output_path or error
    
    def __init__(self, text, voice, custom_voice_path, output_path, tgt_lang):
        super().__init__()
        self.text = text
        self.voice = voice
        self.custom_voice_path = custom_voice_path
        self.output_path = output_path
        self.tgt_lang = tgt_lang
        
    def run(self):
        try:
            success = generate_tts_file(
                self.text,
                self.output_path,
                lang=self.tgt_lang,
                voice=self.voice,
                custom_voice_path=self.custom_voice_path,
                auto_emotion=False
            )
            self.finished.emit(success, self.output_path)
        except Exception as e:
            self.finished.emit(False, str(e))


class ExportWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str) # success, output_path or error
    
    def __init__(self, video_path, segments, translations, voices, params, custom_output_path=None):
        super().__init__()
        self.video_path = video_path
        self.segments = segments
        self.translations = translations
        self.voices = voices
        self.params = params
        self.custom_output_path = custom_output_path
        self.is_cancelled = False
        self.process = None

    def run(self):
        import tempfile
        import shutil
        import subprocess
        
        temp_dir = tempfile.mkdtemp(prefix="dub_export_")
        try:
            self.status.emit("Initializing directories...")
            self.progress.emit(5)
            
            # Setup paths
            ffmpeg_path = get_ffmpeg_cmd()
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            # Extract audio from original video
            orig_audio_path = os.path.join(temp_dir, "orig_audio.wav")
            has_audio = True
            
            self.status.emit("Extracting background audio...")
            self.progress.emit(10)
            self.log.emit("Extracting audio track from original video...")
            cmd_extract = [
                ffmpeg_path, "-y", "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", orig_audio_path
            ]
            
            self.process = subprocess.Popen(cmd_extract, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process.wait()
            
            if not os.path.exists(orig_audio_path):
                has_audio = False
                self.log.emit("⚠️ Original video has no audio track. Generating silent background track...")
                cmd_extract = [
                    ffmpeg_path, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                    "-t", "10", "-acodec", "pcm_s16le", orig_audio_path
                ]
                self.process = subprocess.Popen(cmd_extract, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.process.wait()
                
            # Load original audio samples for echo detection if match_echo is enabled
            orig_all_samples = None
            match_echo = self.params.get("match_echo", False)
            telephone_effect = self.params.get("telephone_effect", False)
            if has_audio and match_echo:
                try:
                    import wave
                    import numpy as np
                    with wave.open(orig_audio_path, "rb") as wav_in:
                        orig_nframes = wav_in.getnframes()
                        orig_bytes = wav_in.readframes(orig_nframes)
                        orig_all_samples = np.frombuffer(orig_bytes, dtype=np.int16)
                except Exception as e:
                    self.log.emit(f"⚠️ Failed to load original audio for echo detection: {e}")
                    
            # Synthesis of all segments
            self.status.emit("Synthesizing speech segments...")
            self.log.emit("Starting TTS voice generation for all segments...")
            
            tts_clips = []
            total_segs = len(self.segments)
            tgt_lang = self.params.get("tgt_lang", "km")
            custom_voice_path = self.params.get("custom_voice_path")
            auto_speed = self.params.get("auto_speed", True)
            
            for idx, seg in enumerate(self.segments):
                if self.is_cancelled:
                    return
                    
                self.status.emit(f"Synthesizing voice {idx+1}/{total_segs}...")
                self.progress.emit(int(15 + (idx / total_segs) * 65))
                
                translated_text = self.translations[idx]
                segment_voice = self.voices[idx]
                
                raw_tts_path = os.path.join(temp_dir, f"raw_tts_{idx}.mp3")
                success = generate_tts_file(translated_text, raw_tts_path, lang=tgt_lang, voice=segment_voice, custom_voice_path=custom_voice_path, auto_emotion=False)
                if not success or not os.path.exists(raw_tts_path):
                    self.log.emit(f"❌ Failed to generate TTS for segment {idx+1}")
                    continue
                    
                # Convert MP3 to raw mono WAV
                raw_wav_path = os.path.join(temp_dir, f"raw_wav_{idx}.wav")
                cmd_conv = [
                    ffmpeg_path, "-y", "-i", raw_tts_path,
                    "-map_metadata", "-1",
                    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", raw_wav_path
                ]
                self.process = subprocess.Popen(cmd_conv, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.process.wait()
                
                # Trim silence
                trimmed_wav_path = os.path.join(temp_dir, f"trimmed_{idx}.wav")
                success_trim = trim_wav_silence(raw_wav_path, trimmed_wav_path, threshold=150)
                if not success_trim or not os.path.exists(trimmed_wav_path):
                    trimmed_wav_path = raw_wav_path
                    
                tts_dur = get_audio_duration(trimmed_wav_path)
                orig_dur = seg["end"] - seg["start"]
                
                # Detect echo
                has_segment_echo = False
                echo_delay = 0.0
                echo_decay = 0.0
                if orig_all_samples is not None and match_echo:
                    try:
                        start_sample = int(seg["start"] * 24000)
                        end_sample = int(seg["end"] * 24000)
                        segment_samples = orig_all_samples[start_sample:end_sample]
                        has_segment_echo, echo_delay, echo_decay = detect_echo(segment_samples, sample_rate=24000)
                    except Exception:
                        pass
                        
                adjusted_tts_path = os.path.join(temp_dir, f"adj_tts_{idx}.wav")
                filters = []
                if auto_speed and orig_dur > 0.1:
                    speed = tts_dur / orig_dur
                    if speed > 1.45:
                        speed = 1.45
                    elif speed < 1.0:
                        speed = 1.0
                    if abs(speed - 1.0) > 0.01:
                        filters.append(f"atempo={speed}")
                        
                if filters:
                    cmd_speed = [
                        ffmpeg_path, "-y", "-i", trimmed_wav_path,
                        "-filter:a", ",".join(filters),
                        "-map_metadata", "-1",
                        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", adjusted_tts_path
                    ]
                    self.process = subprocess.Popen(cmd_speed, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.process.wait()
                else:
                    shutil.copyfile(trimmed_wav_path, adjusted_tts_path)
                    
                final_clip_dur = get_audio_duration(adjusted_tts_path)
                tts_clips.append({
                    "path": adjusted_tts_path,
                    "start": seg["start"],
                    "end": seg["start"] + final_clip_dur,
                    "text": translated_text
                })
                
            # Mix clips using numpy PCM mixer
            self.status.emit("Mixing aligned speech clips...")
            self.progress.emit(80)
            
            video_duration = get_audio_duration(orig_audio_path)
            if video_duration <= 0.0:
                video_duration = 300.0
                
            raw_pcm_bytes = mix_pcm_clips(tts_clips, video_duration, sample_rate=24000)
            
            import wave
            dubbed_speech_wav = os.path.join(temp_dir, "dubbed_speech.wav")
            with wave.open(dubbed_speech_wav, "wb") as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)
                wav_out.setframerate(24000)
                wav_out.writeframes(raw_pcm_bytes)
                
            self.status.emit("Rendering final dubbed video...")
            self.progress.emit(90)
            
            vol_original = self.params.get("vol_original", 0.5)
            vol_dubbed = self.params.get("vol_dubbed", 1.5)
            mute_vocals = self.params.get("mute_vocals", False)
            output_dir = self.params.get("output_dir")
            
            if self.custom_output_path:
                final_output_path = self.custom_output_path
            else:
                video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
                output_filename = f"{video_basename}_dubbed_{tgt_lang}.mp4"
                final_output_path = os.path.join(output_dir, output_filename)
            
            # Ducking
            if has_audio and vol_original > 0.001:
                channels = get_audio_channels(self.video_path)
                is_true_stereo = False
                if channels >= 2:
                    is_true_stereo = check_stereo_difference(self.video_path, ffmpeg_path, temp_dir)
                    
                duck_intervals = []
                for seg in self.segments:
                    start = max(0.0, seg["start"] - 0.1)
                    end = seg["end"] + 0.2
                    duck_intervals.append((start, end))
                    
                if duck_intervals:
                    sum_between = "+".join([f"between(t,{s:.3f},{e:.3f})" for s, e in duck_intervals])
                    vol_filter = f"volume='if({sum_between},{vol_original * 0.05:.3f},{vol_original:.2f})':eval=frame"
                else:
                    vol_filter = f"volume={vol_original:.2f}"
                    
                if mute_vocals and is_true_stereo:
                    # Vocal Removal + BGM Keep:
                    # Phase-Cancellation removes centre-panned vocals.
                    # BGM ducked to 5% during speech, 100% between speech.
                    if duck_intervals:
                        sum_speech = "+".join([f"between(t,{s:.3f},{e:.3f})" for s, e in duck_intervals])
                        bgm_vol_filter = f"volume='if({sum_speech},0.05,1.0)':eval=frame"
                    else:
                        bgm_vol_filter = "volume=1.0"
                    cmd_mix = [
                        ffmpeg_path, "-y", "-i", self.video_path, "-i", dubbed_speech_wav,
                        "-filter_complex", f"[0:a]pan=stereo|c0=c0-c1|c1=c0-c1,{bgm_vol_filter}[a0];[1:a]volume={vol_dubbed:.2f}[a1];[a0][a1]amix=inputs=2:duration=first[out_a]",
                        "-map", "0:v", "-map", "[out_a]", "-c:v", "copy", final_output_path
                    ]
                elif mute_vocals and not is_true_stereo:
                    # Mono audio — cannot phase-cancel vocals, but MIX original audio (ducked to 5%)
                    # with dubbed track so background music is preserved.
                    if duck_intervals:
                        sum_between_mono = "+".join([f"between(t,{s:.3f},{e:.3f})" for s, e in duck_intervals])
                        mono_vol_filter = f"volume='if({sum_between_mono},0.015,{vol_original:.2f})':eval=frame"
                    else:
                        mono_vol_filter = f"volume={vol_original:.2f}"
                    cmd_mix = [
                        ffmpeg_path, "-y", "-i", self.video_path, "-i", dubbed_speech_wav,
                        "-filter_complex", f"[0:a]{mono_vol_filter}[a0];[1:a]volume={vol_dubbed:.2f}[a1];[a0][a1]amix=inputs=2:duration=first[out_a]",
                        "-map", "0:v", "-map", "[out_a]", "-c:v", "copy", final_output_path
                    ]
                else:
                    cmd_mix = [
                        ffmpeg_path, "-y", "-i", self.video_path, "-i", dubbed_speech_wav,
                        "-filter_complex", f"[0:a]{vol_filter}[a0];[1:a]volume={vol_dubbed:.2f}[a1];[a0][a1]amix=inputs=2:duration=first[out_a]",
                        "-map", "0:v", "-map", "[out_a]", "-c:v", "copy", final_output_path
                    ]
            else:
                cmd_mix = [
                    ffmpeg_path, "-y", "-i", self.video_path, "-i", dubbed_speech_wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-shortest", final_output_path
                ]
                
            self.process = subprocess.Popen(cmd_mix, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process.wait()
            
            if os.path.exists(final_output_path):
                self.progress.emit(100)
                self.finished.emit(True, final_output_path)
            else:
                self.finished.emit(False, "Failed to render video with FFmpeg.")
                
        except Exception as e:
            self.finished.emit(False, str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class DubberEditorDialog(QDialog):
    def __init__(self, video_path, segments, translations, params, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
        self.video_path = video_path
        self.segments = segments
        self.translations = translations
        self.params = params
        self.parent_page = parent
        
        self.setWindowTitle("📝 Interactive Video Dubbing Workstation")
        self.resize(1200, 750)
        self.setMinimumSize(1000, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #0d0e12;
                font-family: 'Segoe UI', sans-serif;
                color: #e2e8f0;
            }
            QLabel {
                color: #a0aec0;
                font-size: 12px;
            }
            QTableWidget {
                background-color: #12131a;
                gridline-color: #1e202e;
                border: 1px solid #262938;
                border-radius: 6px;
                color: #e2e8f0;
            }
            QTableWidget::item:selected {
                background-color: #2563eb;
                color: white;
            }
            QLineEdit {
                background-color: #181922;
                border: 1px solid #2d3142;
                border-radius: 4px;
                padding: 4px;
                color: white;
            }
            QLineEdit:focus {
                border: 1px solid #00a2ff;
            }
            QComboBox {
                background-color: #181922;
                border: 1px solid #2d3142;
                border-radius: 4px;
                color: white;
                padding: 2px;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #e2e8f0;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #334155;
            }
            QPushButton:pressed {
                background-color: #0f172a;
            }
        """)
        
        main_lay = QHBoxLayout(self)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #1a1c26; width: 4px; }")
        
        # --- LEFT PANEL ---
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(10)
        
        title_lbl = QLabel("📝 Subtitle & Speech Segment Editor")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #00a2ff; margin-bottom: 5px;")
        left_lay.addWidget(title_lbl)
        
        self.table = QTableWidget(len(self.segments), 6)
        self.table.setHorizontalHeaderLabels(["No.", "Time", "Original Speech", "Translated Speech", "Voice/Style", "Action"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: #181922; color: #a0aec0; border: 1px solid #1e202e; padding: 4px; font-weight: bold; }")
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 200)
        self.table.setColumnWidth(4, 180)
        self.table.setColumnWidth(5, 150)
        
        self.voice_combos = []
        self.text_inputs = []
        
        for idx, seg in enumerate(self.segments):
            self.table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            
            time_str = f"{seg['start']:.2f}s - {seg['end']:.2f}s"
            self.table.setItem(idx, 1, QTableWidgetItem(time_str))
            
            orig_item = QTableWidgetItem(seg['text'])
            orig_item.setFlags(orig_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(idx, 2, orig_item)
            
            trans_input = QLineEdit()
            trans_input.setText(self.translations[idx])
            self.table.setCellWidget(idx, 3, trans_input)
            self.text_inputs.append(trans_input)
            
            voice_combo = QComboBox()
            tgt_lang = self.params.get("tgt_lang", "km")
            voices = VOICES_MAP.get(tgt_lang, [])
            detected_voices = self.params.get("detected_voices", [])
            if idx < len(detected_voices):
                default_voice = detected_voices[idx]
            else:
                default_voice = self.params.get("voice")
            default_idx = 0
            for v_name, v_val in voices:
                voice_combo.addItem(v_name, v_val)
                if v_val == default_voice:
                    default_idx = voice_combo.count() - 1
            
            if tgt_lang == "km":
                try:
                    from settings.manager import load_settings
                    settings = load_settings()
                    saved_voices = settings.get("custom_cloned_voices", [])
                    for v in saved_voices:
                        name_label = f"VoxCPM Clone: {v['name']}"
                        value_val = f"voxcpm-custom|{v['path']}"
                        voice_combo.addItem(name_label, value_val)
                        if value_val == default_voice:
                            default_idx = voice_combo.count() - 1
                except Exception as e:
                    logger.error(f"Failed to load custom voices inside table: {e}")
                    
            voice_combo.setCurrentIndex(default_idx)
            self.table.setCellWidget(idx, 4, voice_combo)
            self.voice_combos.append(voice_combo)
            
            # Action cell containing Orig and Test buttons
            action_widget = QWidget()
            action_lay = QHBoxLayout(action_widget)
            action_lay.setContentsMargins(2, 2, 2, 2)
            action_lay.setSpacing(4)
            
            orig_btn = QPushButton("🔊 Orig")
            orig_btn.setToolTip("Play original speech snippet")
            orig_btn.clicked.connect(lambda checked, i=idx: self.play_original_audio(i))
            
            test_btn = QPushButton("🔊 Test")
            test_btn.setToolTip("Play synthesized Khmer speech snippet")
            test_btn.clicked.connect(lambda checked, i=idx: self.play_segment_audio(i))
            
            action_lay.addWidget(orig_btn)
            action_lay.addWidget(test_btn)
            self.table.setCellWidget(idx, 5, action_widget)
            
        left_lay.addWidget(self.table)
        splitter.addWidget(left_widget)
        
        # Invalidate generated preview if the user edits text or changes voice
        for trans_in in self.text_inputs:
            trans_in.textChanged.connect(self.invalidate_preview)
            trans_in.textChanged.connect(self.save_current_editor_state)
        for voice_comb in self.voice_combos:
            voice_comb.currentIndexChanged.connect(self.invalidate_preview)
            voice_comb.currentIndexChanged.connect(self.save_current_editor_state)
        
        # --- RIGHT PANEL ---
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(10)
        
        self.dubbed_preview_path = None
        
        player_title_row = QHBoxLayout()
        player_title = QLabel("🎥 Video Preview")
        player_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #10b981;")
        
        self.combo_preview_mode = QComboBox()
        self.combo_preview_mode.addItem("Original Video", "orig")
        self.combo_preview_mode.addItem("Dubbed Preview (Generate)", "dubbed")
        self.combo_preview_mode.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #e2e8f0;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.combo_preview_mode.currentIndexChanged.connect(self.on_preview_mode_changed)
        
        player_title_row.addWidget(player_title)
        player_title_row.addStretch()
        player_title_row.addWidget(self.combo_preview_mode)
        right_lay.addLayout(player_title_row)
        
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black; border-radius: 6px; border: 1px solid #262938;")
        self.video_widget.setMinimumWidth(360)
        self.video_widget.setMinimumHeight(240)
        right_lay.addWidget(self.video_widget, 4)
        
        ctrl_row = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶️ Play")
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        ctrl_row.addWidget(self.btn_play_pause)
        
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.sliderMoved.connect(self.set_position)
        ctrl_row.addWidget(self.time_slider)
        
        self.lbl_time = QLabel("00:00 / 00:00")
        ctrl_row.addWidget(self.lbl_time)
        right_lay.addLayout(ctrl_row)
        
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Video Volume:"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(70)
        self.vol_slider.setFixedWidth(120)
        self.vol_slider.valueChanged.connect(self.set_volume)
        vol_row.addWidget(self.vol_slider)
        vol_row.addStretch()
        right_lay.addLayout(vol_row)
        
        right_lay.addSpacing(15)
        
        self.btn_export = QPushButton("🚀 Export Final Video")
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #10b981;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        self.btn_export.clicked.connect(self.export_final_video)
        right_lay.addWidget(self.btn_export)
        
        splitter.addWidget(right_widget)
        
        # Sizing and stretch factors
        splitter.setSizes([750, 450])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        
        # Set minimum widths to prevent panels from collapsing
        left_widget.setMinimumWidth(550)
        right_widget.setMinimumWidth(380)
        
        main_lay.addWidget(splitter)
        
        # Initialize video player
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        
        from PyQt6.QtCore import QUrl
        self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
        self.audio_output.setVolume(0.7)
        
        self.segment_player = QMediaPlayer()
        self.segment_audio_output = QAudioOutput()
        self.segment_player.setAudioOutput(self.segment_audio_output)

    def toggle_playback(self):
        self.sync_play_end_time_ms = None
        from PyQt6.QtMultimedia import QMediaPlayer
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.btn_play_pause.setText("▶️ Play")
        else:
            self.media_player.play()
            self.btn_play_pause.setText("⏸️ Pause")

    def position_changed(self, position):
        self.time_slider.setValue(position)
        self.update_time_label(position, self.media_player.duration())
        
        # Auto-pause at the end of synced segment playback
        if hasattr(self, "sync_play_end_time_ms") and self.sync_play_end_time_ms is not None:
            if position >= self.sync_play_end_time_ms:
                self.media_player.pause()
                self.btn_play_pause.setText("▶️ Play")
                self.sync_play_end_time_ms = None

    def duration_changed(self, duration):
        self.time_slider.setRange(0, duration)

    def set_position(self, position):
        self.sync_play_end_time_ms = None
        self.media_player.setPosition(position)

    def set_volume(self, value):
        self.audio_output.setVolume(value / 100.0)

    def update_time_label(self, position, duration):
        pos_sec = position // 1000
        dur_sec = duration // 1000
        pos_min = pos_sec // 60
        pos_sec = pos_sec % 60
        dur_min = dur_sec // 60
        dur_sec = dur_sec % 60
        self.lbl_time.setText(f"{pos_min:02d}:{pos_sec:02d} / {dur_min:02d}:{dur_sec:02d}")

    def play_original_audio(self, idx):
        # Stop any preview playing on segment_player
        self.segment_player.stop()
        
        seg = self.segments[idx]
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        
        # Add 150ms padding to the end to let the last syllable finish naturally
        self.sync_play_end_time_ms = end_ms + 150
        
        # Seek and play the main video player
        self.media_player.setPosition(start_ms)
        self.media_player.play()
        self.btn_play_pause.setText("⏸️ Pause")

    def play_segment_audio(self, idx):
        text = self.text_inputs[idx].text().strip()
        voice = self.voice_combos[idx].currentData()
        custom_voice_path = self.params.get("custom_voice_path")
        tgt_lang = self.params.get("tgt_lang", "km")
        
        if not text:
            QMessageBox.warning(self, "Warning", "Please input speech text first.")
            return
            
        cell_widget = self.table.cellWidget(idx, 5)
        buttons = cell_widget.findChildren(QPushButton)
        orig_btn = buttons[0]
        test_btn = buttons[1]
        
        orig_btn.setEnabled(False)
        test_btn.setEnabled(False)
        test_btn.setText("⏳ ...")
        
        import tempfile
        temp_dir = tempfile.gettempdir()
        out_path = os.path.join(temp_dir, f"segment_preview_{idx}.mp3")
        
        self.preview_worker = SegmentTTSWorker(text, voice, custom_voice_path, out_path, tgt_lang)
        self.preview_worker.finished.connect(lambda success, path, i=idx: self.on_preview_finished(i, success, path))
        self.preview_worker.start()

    def on_preview_finished(self, idx, success, path):
        cell_widget = self.table.cellWidget(idx, 5)
        buttons = cell_widget.findChildren(QPushButton)
        orig_btn = buttons[0]
        test_btn = buttons[1]
        
        orig_btn.setEnabled(True)
        test_btn.setEnabled(True)
        test_btn.setText("🔊 Test")
        
        if success and os.path.exists(path):
            from PyQt6.QtCore import QUrl
            self.segment_player.stop()
            self.segment_player.setSource(QUrl.fromLocalFile(path))
            self.segment_audio_output.setVolume(1.0)
            self.segment_player.play()
        else:
            QMessageBox.warning(self, "Error", f"Failed to synthesize segment speech: {path}")

    def export_final_video(self):
        self.media_player.pause()
        self.btn_play_pause.setText("▶️ Play")
        
        translations = [inp.text().strip() for inp in self.text_inputs]
        voices = [combo.currentData() for combo in self.voice_combos]
        
        self.params["match_echo"] = False
        self.params["telephone_effect"] = False
        
        if self.parent_page:
            try:
                self.params["vol_original"] = self.parent_page.combo_original_vol.currentData()
                self.params["vol_dubbed"] = self.parent_page.combo_dubbed_vol.currentData()
                self.params["auto_speed"] = self.parent_page.chk_auto_speed.isChecked()
            except Exception as e:
                logger.error(f"Failed to dynamically load settings from parent page: {e}")
        
        from PyQt6.QtWidgets import QProgressDialog
        self.progress_dialog = QProgressDialog("Synthesizing edited voice tracks and rendering video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Rendering Video")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        
        self.export_worker = ExportWorker(self.video_path, self.segments, translations, voices, self.params)
        self.export_worker.progress.connect(self.progress_dialog.setValue)
        self.export_worker.status.connect(self.progress_dialog.setLabelText)
        self.export_worker.finished.connect(self.on_export_finished)
        
        self.progress_dialog.canceled.connect(self.cancel_export)
        self.export_worker.start()

    def cancel_export(self):
        if hasattr(self, "export_worker") and self.export_worker.isRunning():
            self.export_worker.is_cancelled = True
            if self.export_worker.process:
                try:
                    self.export_worker.process.terminate()
                except Exception:
                    pass
            QMessageBox.information(self, "Cancelled", "Video rendering was cancelled.")

    def on_export_finished(self, success, path_or_err):
        if hasattr(self, "export_worker") and self.export_worker.is_cancelled:
            return
        self.progress_dialog.close()
        if success:
            QMessageBox.information(self, "Success", f"🎉 Dubbed video exported successfully to:\n{path_or_err}")
            try:
                import os
                os.startfile(os.path.dirname(path_or_err))
            except Exception:
                pass
            self.accept()
        else:
            QMessageBox.critical(self, "Export Failed", f"An error occurred during video rendering:\n{path_or_err}")

    def on_preview_mode_changed(self, index):
        mode = self.combo_preview_mode.currentData()
        from PyQt6.QtCore import QUrl
        
        if mode == "orig":
            current_pos = self.media_player.position()
            self.media_player.stop()
            self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
            self.media_player.setPosition(current_pos)
            self.media_player.play()
            self.btn_play_pause.setText("⏸️ Pause")
        elif mode == "dubbed":
            if hasattr(self, "dubbed_preview_path") and self.dubbed_preview_path and os.path.exists(self.dubbed_preview_path):
                current_pos = self.media_player.position()
                self.media_player.stop()
                self.media_player.setSource(QUrl.fromLocalFile(self.dubbed_preview_path))
                self.media_player.setPosition(current_pos)
                self.media_player.play()
                self.btn_play_pause.setText("⏸️ Pause")
            else:
                self.generate_dubbed_preview()

    def generate_dubbed_preview(self):
        self.media_player.pause()
        self.btn_play_pause.setText("▶️ Play")
        
        translations = [inp.text().strip() for inp in self.text_inputs]
        voices = [combo.currentData() for combo in self.voice_combos]
        
        self.params["match_echo"] = False
        self.params["telephone_effect"] = False
        
        if self.parent_page:
            try:
                self.params["vol_original"] = self.parent_page.combo_original_vol.currentData()
                self.params["vol_dubbed"] = self.parent_page.combo_dubbed_vol.currentData()
                self.params["auto_speed"] = self.parent_page.chk_auto_speed.isChecked()
            except Exception as e:
                logger.error(f"Failed to dynamically load settings from parent page: {e}")
                
        import tempfile
        import uuid
        temp_dir = tempfile.gettempdir()
        self.temp_preview_path = os.path.join(temp_dir, f"dub_preview_{uuid.uuid4().hex[:8]}.mp4")
        self.preview_restore_pos = self.media_player.position()
        
        from PyQt6.QtWidgets import QProgressDialog
        self.preview_progress_dialog = QProgressDialog("Generating dubbed preview video...", "Cancel", 0, 100, self)
        self.preview_progress_dialog.setWindowTitle("Generating Preview")
        self.preview_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.preview_progress_dialog.setMinimumDuration(0)
        self.preview_progress_dialog.setValue(0)
        
        self.preview_export_worker = ExportWorker(
            self.video_path, self.segments, translations, voices, self.params,
            custom_output_path=self.temp_preview_path
        )
        self.preview_export_worker.progress.connect(self.preview_progress_dialog.setValue)
        self.preview_export_worker.status.connect(self.preview_progress_dialog.setLabelText)
        self.preview_export_worker.finished.connect(self.on_preview_generation_finished)
        
        self.preview_progress_dialog.canceled.connect(self.cancel_preview_generation)
        self.preview_export_worker.start()

    def cancel_preview_generation(self):
        if hasattr(self, "preview_export_worker") and self.preview_export_worker.isRunning():
            self.preview_export_worker.is_cancelled = True
            if self.preview_export_worker.process:
                try:
                    self.preview_export_worker.process.terminate()
                except Exception:
                    pass
        self.combo_preview_mode.blockSignals(True)
        self.combo_preview_mode.setCurrentIndex(0)
        self.combo_preview_mode.blockSignals(False)

    def on_preview_generation_finished(self, success, path_or_err):
        if hasattr(self, "preview_export_worker") and self.preview_export_worker.is_cancelled:
            return
        self.preview_progress_dialog.close()
        from PyQt6.QtCore import QUrl
        if success and os.path.exists(self.temp_preview_path):
            self.dubbed_preview_path = self.temp_preview_path
            
            self.media_player.stop()
            self.media_player.setSource(QUrl.fromLocalFile(self.dubbed_preview_path))
            
            restore_pos = getattr(self, "preview_restore_pos", 0)
            self.media_player.setPosition(restore_pos)
            self.media_player.play()
            self.btn_play_pause.setText("⏸️ Pause")
            
            self.combo_preview_mode.blockSignals(True)
            self.combo_preview_mode.setItemText(1, "Dubbed Preview (Active)")
            self.combo_preview_mode.setCurrentIndex(1)
            self.combo_preview_mode.blockSignals(False)
        else:
            QMessageBox.critical(self, "Preview Failed", f"Failed to generate dubbed preview:\n{path_or_err}")
            self.combo_preview_mode.blockSignals(True)
            self.combo_preview_mode.setCurrentIndex(0)
            self.combo_preview_mode.blockSignals(False)

    def invalidate_preview(self):
        if hasattr(self, "dubbed_preview_path") and self.dubbed_preview_path:
            try:
                if os.path.exists(self.dubbed_preview_path):
                    os.remove(self.dubbed_preview_path)
            except Exception:
                pass
            self.dubbed_preview_path = None
            
        self.combo_preview_mode.blockSignals(True)
        self.combo_preview_mode.setItemText(1, "Dubbed Preview (Generate)")
        if self.combo_preview_mode.currentIndex() == 1:
            self.combo_preview_mode.setCurrentIndex(0)
            current_pos = self.media_player.position()
            self.media_player.stop()
            from PyQt6.QtCore import QUrl
            self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
            self.media_player.setPosition(current_pos)
            self.media_player.play()
            self.btn_play_pause.setText("⏸️ Pause")
        self.combo_preview_mode.blockSignals(False)

    def save_current_editor_state(self):
        try:
            translations = [inp.text().strip() for inp in self.text_inputs]
            voices = [combo.currentData() for combo in self.voice_combos]
            
            self.params["match_echo"] = False
            self.params["telephone_effect"] = False
            
            # Read the existing resume state and update it
            resume_data = load_resume_state()
            if resume_data:
                resume_data["translations"] = translations
                resume_data["params"]["detected_voices"] = voices
                resume_data["params"]["match_echo"] = self.params["match_echo"]
                resume_data["params"]["telephone_effect"] = self.params["telephone_effect"]
                resume_data["step"] = "interactive_edited"
                save_resume_state(resume_data)
                
            # Update global dub cache with current edits
            try:
                cache = load_dub_cache()
                cache[self.video_path] = {
                    "segments": self.segments,
                    "translations": translations,
                    "voices": voices,
                    "params": self.params
                }
                save_dub_cache(cache)
            except Exception as ce:
                logger.error(f"Failed to update global dub cache: {ce}")
        except Exception as e:
            logger.error(f"Failed to save current editor state: {e}")

    def closeEvent(self, event):
        self.media_player.stop()
        self.segment_player.stop()
        
        try:
            resume_data = load_resume_state()
            if resume_data:
                temp_dir = resume_data.get("temp_dir")
                if temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                clear_resume_state()
        except Exception as e:
            logger.error(f"Failed to cleanup temp dir on dialog close: {e}")
            
        super().closeEvent(event)


class DragDropListWidget(QListWidget):
    filesDropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.reset_style()

    def reset_style(self):
        self.setStyleSheet("""
            QListWidget {
                background-color: #13141f;
                border: 2px dashed #262938;
                border-radius: 8px;
                color: #e2e8f0;
                padding: 5px;
            }
        """)

    def highlight_style(self):
        self.setStyleSheet("""
            QListWidget {
                background-color: #172554;
                border: 2px dashed #2563eb;
                border-radius: 8px;
                color: #e2e8f0;
                padding: 5px;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.highlight_style()

    def dragLeaveEvent(self, event):
        self.reset_style()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        self.reset_style()
        urls = event.mimeData().urls()
        file_paths = []
        for url in urls:
            path = url.toLocalFile()
            if os.path.exists(path):
                file_paths.append(path)
        if file_paths:
            self.filesDropped.emit(file_paths)
        event.acceptProposedAction()


class VideoDubberPage(QWidget):
    def __init__(self, dashboard=None):
        super().__init__()
        self.dashboard = dashboard
        self.jobs = []            # list of dicts: {id, type, name, inputs, output, details, status, progress, dubbing_params}
        self.active_workers = {}  # dict of job_id: worker
        self.finished_workers = []  # preservation list to prevent GC segfaults
        self.max_parallel = 10
        self.running = False
        self.init_ui()
        self.load_dubber_session()
        
        # Check for resume state after UI is ready
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1000, self.check_and_prompt_resume)

    def init_ui(self):
        # Apply dark premium styles
        self.setStyleSheet("""
            QWidget {
                background-color: #0f1015;
                font-family: 'Segoe UI', sans-serif;
                color: #e2e8f0;
            }
            QGroupBox {
                background-color: #12131a;
                border: 1px solid #262938;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 13px;
                color: #ffffff;
                padding: 14px 10px 10px 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0px 5px;
            }
            QLabel {
                color: #a0aec0;
                font-size: 12px;
            }
            QComboBox {
                background-color: #1e202f;
                color: white;
                border: 1px solid #2d3142;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QLineEdit {
                background-color: #1e202f;
                border: 1px solid #2d3142;
                border-radius: 4px;
                padding: 4px 6px;
                color: white;
                min-height: 24px;
            }
            QLineEdit:focus {
                border: 1px solid #00a2ff;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #1e202e; width: 4px; }")

        # ----------------- LEFT PANEL (INPUTS & SETTINGS) -----------------
        left_widget = QWidget()
        left_widget.setMinimumWidth(430)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Config tab widget or scroll area
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 5, 0)
        scroll_layout.setSpacing(10)

        # Video list group
        list_group = QGroupBox("🎥 Video Dubber List (Drag files or Add)")
        list_group_lay = QVBoxLayout(list_group)
        list_group_lay.setSpacing(8)
        list_group_lay.setContentsMargins(8, 12, 8, 8)

        self.list_dub_files = DragDropListWidget()
        self.list_dub_files.setFixedHeight(140)
        self.list_dub_files.filesDropped.connect(self.add_dub_files)
        self.list_dub_files.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_dub_files.customContextMenuRequested.connect(self.show_list_context_menu)
        list_group_lay.addWidget(self.list_dub_files)

        # List controls row
        list_ctrl_lay = QHBoxLayout()
        self.btn_add_files = QPushButton("➕ Add File")
        self.btn_add_files.clicked.connect(self.browse_dub_files)
        self.btn_add_folder = QPushButton("📁 Add Folder")
        self.btn_add_folder.clicked.connect(self.browse_dub_folder)
        self.btn_sort_files = QPushButton("🔀 Sort A-Z/1-100")
        self.btn_sort_files.clicked.connect(self.sort_dub_files)
        self.btn_remove_file = QPushButton("🗑️ Remove")
        self.btn_remove_file.clicked.connect(self.remove_dub_file)
        self.btn_clear_files = QPushButton("🧹 Clear")
        self.btn_clear_files.clicked.connect(self.clear_dub_files)
        
        list_ctrl_lay.addWidget(self.btn_add_files)
        list_ctrl_lay.addWidget(self.btn_add_folder)
        list_ctrl_lay.addWidget(self.btn_sort_files)
        list_ctrl_lay.addWidget(self.btn_remove_file)
        list_ctrl_lay.addWidget(self.btn_clear_files)
        list_group_lay.addLayout(list_ctrl_lay)
        scroll_layout.addWidget(list_group)

        # Output Directory Group
        out_group = QGroupBox("📁 Output Settings")
        out_group_lay = QVBoxLayout(out_group)
        out_group_lay.setSpacing(6)
        out_group_lay.setContentsMargins(8, 12, 8, 8)

        out_group_lay.addWidget(QLabel("Output Directory:"))
        out_dir_row = QHBoxLayout()
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setReadOnly(True)
        self.txt_output_dir.setText(get_default_export_dir())
        self.btn_browse_out = QPushButton("📁")
        self.btn_browse_out.setFixedWidth(30)
        self.btn_browse_out.clicked.connect(self.browse_output_dir)
        out_dir_row.addWidget(self.txt_output_dir)
        out_dir_row.addWidget(self.btn_browse_out)
        out_group_lay.addLayout(out_dir_row)
        scroll_layout.addWidget(out_group)

        # Language settings
        lang_group = QGroupBox("Language Settings")
        lang_grid = QGridLayout(lang_group)
        lang_grid.setSpacing(8)
        lang_grid.setContentsMargins(8, 12, 8, 8)

        lang_grid.addWidget(QLabel("Target Language:"), 0, 0)
        self.combo_target_lang = QComboBox()
        self.combo_target_lang.addItem("Khmer (ខ្មែរ)", "km")
        self.combo_target_lang.addItem("English", "en")
        self.combo_target_lang.addItem("Thai (ไทย)", "th")
        self.combo_target_lang.addItem("Vietnamese (Tiếng Việt)", "vi")
        self.combo_target_lang.addItem("Chinese (中文)", "zh-cn")
        self.combo_target_lang.addItem("Japanese (日本語)", "ja")
        self.combo_target_lang.addItem("Korean (한국어)", "ko")
        self.combo_target_lang.addItem("French (Français)", "fr")
        self.combo_target_lang.addItem("Spanish (Español)", "es")
        lang_grid.addWidget(self.combo_target_lang, 0, 1)

        lang_grid.addWidget(QLabel("Target Voice:"), 1, 0)
        voice_row = QHBoxLayout()
        self.combo_voice = QComboBox()
        self.btn_delete_cloned_voice = QPushButton("🗑")
        self.btn_delete_cloned_voice.setFixedWidth(42)
        self.btn_delete_cloned_voice.setStyleSheet("""
            QPushButton {
                background-color: #1e1b4b;
                color: #ef4444;
                border: 1px solid #ef4444;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 0px;
                min-height: 28px;
                max-height: 28px;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: white;
            }
            QPushButton:disabled {
                background-color: #27272a;
                color: #71717a;
                border: 1px solid #3f3f46;
            }
        """)
        self.btn_delete_cloned_voice.setToolTip("Delete selected custom cloned voice")
        self.btn_delete_cloned_voice.clicked.connect(self.delete_custom_voice_profile)
        self.btn_manage_local_server = QPushButton("⚙️ Local Server")
        self.btn_manage_local_server.setStyleSheet("background-color: #0f172a; color: white; border-radius: 4px; padding: 4px;")
        self.btn_manage_local_server.clicked.connect(self.open_local_server_manager)
        voice_row.addWidget(self.combo_voice, 1)
        voice_row.addWidget(self.btn_delete_cloned_voice)
        voice_row.addWidget(self.btn_manage_local_server)
        lang_grid.addLayout(voice_row, 1, 1)

        # Custom Voice Reference Row (hidden by default)
        self.lbl_custom_voice = QLabel("Voice Sample (WAV/MP3):")
        self.custom_voice_widget = QWidget()
        custom_voice_lay = QHBoxLayout(self.custom_voice_widget)
        custom_voice_lay.setContentsMargins(0, 0, 0, 0)
        self.txt_custom_voice_path = QLineEdit()
        self.txt_custom_voice_path.setPlaceholderText("Select 5-15s clean voice recording...")
        
        # Load saved custom voice path if it exists
        try:
            from settings.manager import load_settings
            settings = load_settings()
            saved_voice_path = settings.get("custom_voice_path", "")
            if saved_voice_path and os.path.exists(saved_voice_path):
                self.txt_custom_voice_path.setText(saved_voice_path)
        except Exception as e:
            logger.error(f"Failed to load custom_voice_path: {e}")
            
        self.txt_custom_voice_path.textChanged.connect(self.save_custom_voice_path)
        
        self.btn_browse_custom_voice = QPushButton("📁")
        self.btn_browse_custom_voice.setFixedWidth(30)
        self.btn_browse_custom_voice.clicked.connect(self.browse_custom_voice_file)
        
        self.btn_save_custom_voice = QPushButton("💾 Save")
        self.btn_save_custom_voice.setFixedWidth(55)
        self.btn_save_custom_voice.setStyleSheet("background-color: #1e293b; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_save_custom_voice.clicked.connect(self.save_custom_voice_profile)
        
        custom_voice_lay.addWidget(self.txt_custom_voice_path)
        custom_voice_lay.addWidget(self.btn_browse_custom_voice)
        custom_voice_lay.addWidget(self.btn_save_custom_voice)
        
        lang_grid.addWidget(self.lbl_custom_voice, 2, 0)
        lang_grid.addWidget(self.custom_voice_widget, 2, 1)
        
        self.lbl_custom_voice.setVisible(False)
        self.custom_voice_widget.setVisible(False)

        lang_grid.addWidget(QLabel("Source Language:"), 3, 0)
        self.combo_source_lang = QComboBox()
        self.combo_source_lang.addItem("Auto Detect", "Auto Detect")
        self.combo_source_lang.addItem("English", "en")
        self.combo_source_lang.addItem("Khmer (ខ្មែរ)", "km")
        self.combo_source_lang.addItem("Chinese (中文)", "zh")
        self.combo_source_lang.addItem("Japanese (日本語)", "ja")
        self.combo_source_lang.addItem("Thai (ไทย)", "th")
        self.combo_source_lang.addItem("French (Français)", "fr")
        self.combo_source_lang.addItem("Spanish (Español)", "es")
        lang_grid.addWidget(self.combo_source_lang, 3, 1)
        
        lang_grid.addWidget(QLabel("Translation Engine:"), 4, 0)
        self.combo_trans_engine = QComboBox()
        self.combo_trans_engine.addItem("Google Translate (Free)", "google")
        self.combo_trans_engine.addItem("Gemini AI (High Quality)", "gemini")
        self.combo_trans_engine.currentIndexChanged.connect(self.on_translation_engine_changed)
        lang_grid.addWidget(self.combo_trans_engine, 4, 1)
        
        self.lbl_gemini_model = QLabel("Gemini Model:")
        lang_grid.addWidget(self.lbl_gemini_model, 5, 0)
        self.combo_gemini_model = QComboBox()
        self.combo_gemini_model.addItem("Gemini 3.1 Flash-Lite (Low Latency)", "gemini-3.1-flash-lite")
        self.combo_gemini_model.addItem("Gemini 2.5 Flash (Recommended)", "gemini-2.5-flash")
        self.combo_gemini_model.addItem("Gemini 2.5 Pro (High Quality)", "gemini-2.5-pro")
        self.combo_gemini_model.addItem("Gemini 1.5 Flash (Legacy)", "gemini-1.5-flash")
        self.combo_gemini_model.addItem("Gemini 1.5 Pro (Legacy)", "gemini-1.5-pro")
        self.combo_gemini_model.currentIndexChanged.connect(self.on_gemini_model_changed)
        lang_grid.addWidget(self.combo_gemini_model, 5, 1)
        
        self.lbl_gemini_key = QLabel("Gemini API Key:")
        lang_grid.addWidget(self.lbl_gemini_key, 6, 0)
        
        self.container_gemini = QWidget()
        gemini_lay = QHBoxLayout(self.container_gemini)
        gemini_lay.setContentsMargins(0, 0, 0, 0)
        self.txt_gemini_api_key = QLineEdit()
        self.txt_gemini_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_gemini_api_key.setPlaceholderText("Enter Gemini API Key(s). Separate multiple keys with commas, semicolons, or newlines.")
        self.txt_gemini_api_key.setFixedHeight(28)
        self.txt_gemini_api_key.setStyleSheet("font-size: 12px; padding: 2px 4px;")
        
        self.btn_check_gemini = QPushButton("Verify")
        self.btn_check_gemini.setFixedWidth(65)
        self.btn_check_gemini.setFixedHeight(28)
        self.btn_check_gemini.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #3b82f6;
                border: 1px solid #3b82f6;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3b82f6;
                color: white;
            }
            QPushButton:disabled {
                background-color: #27272a;
                color: #71717a;
                border: 1px solid #3f3f46;
            }
        """)
        self.btn_check_gemini.setToolTip("Test Gemini API key validity")
        self.btn_check_gemini.clicked.connect(self.verify_gemini_key)
        
        self.btn_manage_gemini = QPushButton("Manage Keys")
        self.btn_manage_gemini.setFixedWidth(110)
        self.btn_manage_gemini.setFixedHeight(28)
        self.btn_manage_gemini.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #10b981;
                border: 1px solid #10b981;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #10b981;
                color: white;
            }
        """)
        self.btn_manage_gemini.setToolTip("Add/Remove/Verify multiple Gemini API keys in a dialog")
        self.btn_manage_gemini.clicked.connect(self.show_gemini_key_manager)
        
        gemini_lay.addWidget(self.txt_gemini_api_key, 1)
        gemini_lay.addWidget(self.btn_check_gemini)
        gemini_lay.addWidget(self.btn_manage_gemini)
        
        try:
            from settings.manager import load_settings
            settings = load_settings()
            self.txt_gemini_api_key.setText(settings.get("gemini_api_key", ""))
            
            has_key = bool(settings.get("gemini_api_key", "").strip())
            default_engine = "gemini" if has_key else "google"
            engine = settings.get("translation_engine", default_engine)
            idx = self.combo_trans_engine.findData(engine)
            if idx >= 0:
                self.combo_trans_engine.setCurrentIndex(idx)
                
            model_name = settings.get("gemini_model", "gemini-2.5-flash")
            idx_model = self.combo_gemini_model.findData(model_name)
            if idx_model >= 0:
                self.combo_gemini_model.setCurrentIndex(idx_model)
        except Exception as e:
            logger.error(f"Failed to load translation settings in dubber UI: {e}")
            
        self.txt_gemini_api_key.textChanged.connect(self.save_gemini_api_key)
        lang_grid.addWidget(self.container_gemini, 6, 1)
        
        self.on_translation_engine_changed()
        
        self.combo_target_lang.currentIndexChanged.connect(self.update_voice_dropdown)
        self.combo_voice.currentIndexChanged.connect(self.on_voice_changed)
        self.update_voice_dropdown()
        scroll_layout.addWidget(lang_group)

        # Whisper Settings
        model_group = QGroupBox("Whisper Model Settings")
        model_grid = QGridLayout(model_group)
        model_grid.setSpacing(8)
        model_grid.setContentsMargins(8, 12, 8, 8)

        model_grid.addWidget(QLabel("Model Size:"), 0, 0)
        model_size_layout = QHBoxLayout()
        self.combo_model_size = QComboBox()
        self.combo_model_size.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.combo_model_size.setCurrentText("medium")
        model_size_layout.addWidget(self.combo_model_size, 1)
        
        self.btn_download_model = QPushButton("Download")
        self.btn_download_model.clicked.connect(self.start_model_download)
        model_size_layout.addWidget(self.btn_download_model)
        model_grid.addLayout(model_size_layout, 0, 1)
        
        self.combo_model_size.currentTextChanged.connect(self.update_model_status)
        self.combo_model_size.currentTextChanged.connect(lambda: self.save_dubber_session())
        self.update_model_status()

        model_grid.addWidget(QLabel("Acceleration:"), 1, 0)
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cuda", "cpu"])
        self.combo_device.setCurrentText("cuda")
        self.combo_device.currentTextChanged.connect(lambda: self.save_dubber_session())
        model_grid.addWidget(self.combo_device, 1, 1)
        scroll_layout.addWidget(model_group)

        # Audio options
        audio_group = QGroupBox("Audio Overlay Options")
        audio_grid = QGridLayout(audio_group)
        audio_grid.setSpacing(8)
        audio_grid.setContentsMargins(8, 12, 8, 8)

        audio_grid.addWidget(QLabel("Original Audio Vol:"), 0, 0)
        self.combo_original_vol = QComboBox()
        self.combo_original_vol.addItem("Mute (0%)", 0.0)
        self.combo_original_vol.addItem("Quiet (5%)", 0.05)
        self.combo_original_vol.addItem("Ducked (15%)", 0.15)
        self.combo_original_vol.addItem("Low (25%)", 0.25)
        self.combo_original_vol.addItem("Medium (50%)", 0.5)
        self.combo_original_vol.addItem("Normal (100%)", 1.0)
        self.combo_original_vol.setCurrentIndex(0)
        self.combo_original_vol.currentIndexChanged.connect(self.save_dubber_session)
        audio_grid.addWidget(self.combo_original_vol, 0, 1)

        audio_grid.addWidget(QLabel("Dubbed Voice Vol:"), 1, 0)
        self.combo_dubbed_vol = QComboBox()
        self.combo_dubbed_vol.addItem("Extreme Boost (300%)", 3.0)
        self.combo_dubbed_vol.addItem("Double Volume (200%)", 2.0)
        self.combo_dubbed_vol.addItem("Boosted (150%)", 1.5)
        self.combo_dubbed_vol.addItem("Normal (100%)", 1.0)
        self.combo_dubbed_vol.addItem("Soft (75%)", 0.75)
        self.combo_dubbed_vol.setCurrentIndex(0)
        self.combo_dubbed_vol.currentIndexChanged.connect(self.save_dubber_session)
        audio_grid.addWidget(self.combo_dubbed_vol, 1, 1)

        self.chk_auto_speed = QCheckBox("Speed Match (Fast Dubbing Speech)")
        self.chk_auto_speed.setChecked(True)
        audio_grid.addWidget(self.chk_auto_speed, 2, 0, 1, 2)

        self.chk_mute_vocals = QCheckBox("Vocal Removal 100% (Keep Background Music)")
        self.chk_mute_vocals.setChecked(True)
        self.chk_mute_vocals.setToolTip(
            "Stereo: Uses Phase-Cancellation to remove centre-panned vocals and keeps BGM at full volume (100% mute).\n"
            "Mono: Cannot separate vocals from BGM — mutes all original audio and uses only the dubbed track (100% mute)."
        )
        audio_grid.addWidget(self.chk_mute_vocals, 3, 0, 1, 2)

        scroll_layout.addWidget(audio_group)

        left_scroll.setWidget(scroll_content)
        left_layout.addWidget(left_scroll, 2)

        # Log Area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(120)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #13141f;
                border: 1px solid #262938;
                border-radius: 6px;
                color: #a0aec0;
                font-family: Consolas, Monaco, monospace;
                font-size: 11px;
            }
        """)
        left_layout.addWidget(self.log_area, 1)

        # Action Buttons
        self.btn_dub_now = QPushButton("🎙️ Dub All Now")
        self.btn_dub_now.setProperty("class", "btn-blue")
        self.btn_dub_now.clicked.connect(self.dub_all_now)
        
        self.btn_edit_dub = QPushButton("📝 Edit & Dub (Interactive)")
        self.btn_edit_dub.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                font-size: 13px;
                border: none;
            }
            QPushButton:hover {
                background-color: #10b981;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        self.btn_edit_dub.clicked.connect(self.edit_dub_interactive)
        
        self.btn_queue_job = QPushButton("📥 Queue Dubbing Job")
        self.btn_queue_job.clicked.connect(self.queue_dubbing_job)
        
        left_layout.addWidget(self.btn_dub_now)
        left_layout.addWidget(self.btn_edit_dub)
        left_layout.addWidget(self.btn_queue_job)
        splitter.addWidget(left_widget)

        # ----------------- RIGHT PANEL (TASK QUEUE) -----------------
        right_panel = QFrame()
        right_panel.setObjectName("right_panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Queue Header
        q_header = QHBoxLayout()
        q_title = QLabel("📥 Batch Processing Task Queue")
        q_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00a2ff;")
        self.lbl_queue_status = QLabel("Idle")
        self.lbl_queue_status.setStyleSheet("""
            color: #a0aec0;
            font-size: 11px;
            font-weight: bold;
            background-color: #1a1c26;
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid #2d3748;
        """)
        q_header.addWidget(q_title)
        q_header.addStretch()
        q_header.addWidget(self.lbl_queue_status)
        right_layout.addLayout(q_header)

        # Control Buttons
        ctrls_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶️ Start")
        self.btn_start.clicked.connect(self.start_queue)
        
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.clicked.connect(self.pause_queue)
        self.btn_pause.setEnabled(False)
        
        self.btn_remove = QPushButton("🗑️ Remove Selected")
        self.btn_remove.clicked.connect(self.remove_selected)
        
        self.btn_clear_completed = QPushButton("🧹 Clear Done")
        self.btn_clear_completed.clicked.connect(self.clear_completed)

        ctrls_layout.addWidget(self.btn_start)
        ctrls_layout.addWidget(self.btn_pause)
        ctrls_layout.addWidget(self.btn_remove)
        ctrls_layout.addWidget(self.btn_clear_completed)
        ctrls_layout.addSpacing(15)

        lbl_parallel = QLabel("Parallel Limit:")
        lbl_parallel.setStyleSheet("color: #a0aec0; font-size: 12px; font-weight: bold;")
        self.spin_parallel = QSpinBox()
        self.spin_parallel.setRange(1, 999)
        self.spin_parallel.setValue(self.max_parallel)
        self.spin_parallel.setFixedWidth(75)
        self.spin_parallel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_parallel.valueChanged.connect(self.on_max_parallel_changed)
        
        ctrls_layout.addWidget(lbl_parallel)
        ctrls_layout.addWidget(self.spin_parallel)
        ctrls_layout.addStretch()
        right_layout.addLayout(ctrls_layout)

        # Queue Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Task Name", "Type", "Source Media", "Operation Details", "Status", "Progress"
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        right_layout.addWidget(self.table, 1)

        # Overall Progress
        overall_lay = QHBoxLayout()
        self.lbl_overall = QLabel("Overall progress: 0 / 0 jobs completed")
        self.lbl_overall.setStyleSheet("color: #a0aec0; font-weight: bold; font-size: 11px;")
        
        self.progress_overall = QProgressBar()
        self.progress_overall.setValue(0)
        self.progress_overall.setFixedHeight(12)
        self.progress_overall.setStyleSheet("""
            QProgressBar {
                background-color: #1a1b23;
                border: 1px solid #2d3748;
                border-radius: 5px;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #005ea2, stop:1 #00a2ff);
                border-radius: 4px;
            }
        """)
        overall_lay.addWidget(self.lbl_overall)
        overall_lay.addWidget(self.progress_overall, 1)
        right_layout.addLayout(overall_lay)

        splitter.addWidget(right_panel)
        splitter.setSizes([450, 750])
        layout.addWidget(splitter)
        self.apply_theme()

        # Set backwards compatible properties
        self.lbl_status = self.lbl_queue_status
        self.progress_bar = self.progress_overall

    def apply_theme(self):
        btn_style = """
            QPushButton {
                background-color: #1f2029;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 6px 12px;
                color: #e2e8f0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2d3748;
            }
            QPushButton:pressed {
                background-color: #1a1b23;
            }
        """
        blue_btn_style = """
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 5px;
                padding: 7px 14px;
                color: white;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #00a2ff;
            }
            QPushButton:pressed {
                background-color: #005ea2;
            }
        """
        for button in self.findChildren(QPushButton):
            if button.property("class") == "btn-blue":
                button.setStyleSheet(blue_btn_style)
            elif button.text() == "📁":
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #1f2029;
                        border: 1px solid #2d3748;
                        border-radius: 4px;
                        color: #e2e8f0;
                        font-weight: bold;
                    }
                """)
            else:
                button.setStyleSheet(btn_style)

        spin_style = """
            QSpinBox {
                background-color: #1f2029;
                border: 1px solid #2d3748;
                border-radius: 4px;
                color: white;
                padding: 0px 6px 3px 6px;
                height: 28px;
            }
        """
        self.spin_parallel.setStyleSheet(spin_style)

        checkbox_style = """
            QCheckBox {
                color: #e2e8f0;
                font-size: 11px;
            }
        """
        for cb in self.findChildren(QCheckBox):
            cb.setStyleSheet(checkbox_style)

    # ----------------- SIDEBAR FILE LIST HANDLERS -----------------
    def add_dub_files(self, file_paths: list):
        valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        expanded = []
        for path in file_paths:
            if os.path.isdir(path):
                # Automatically create a _Dub folder inside the added folder
                # and set the output directory to it.
                try:
                    folder_name = os.path.basename(os.path.abspath(path))
                    dub_folder_name = f"{folder_name}_Dub"
                    dub_dir_path = os.path.join(path, dub_folder_name)
                    os.makedirs(dub_dir_path, exist_ok=True)
                    self.txt_output_dir.setText(os.path.abspath(dub_dir_path))
                except Exception as e:
                    logger.error(f"Failed to create auto output folder for {path}: {e}")

                base_dir = os.path.dirname(path)
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        expanded.append((file_path, base_dir))
            else:
                expanded.append((path, None))

        for path, base_dir in expanded:
            ext = os.path.splitext(path)[1].lower()
            if ext in valid_exts and os.path.exists(path):
                # Check duplicates in list
                already_in = False
                for idx in range(self.list_dub_files.count()):
                    if self.list_dub_files.item(idx).toolTip() == path:
                        already_in = True
                        break
                if not already_in:
                    item = QListWidgetItem(os.path.basename(path))
                    item.setToolTip(path)
                    if base_dir:
                        item.setData(Qt.ItemDataRole.UserRole, base_dir)
                    self.list_dub_files.addItem(item)
        
        # Auto Sort list A-Z / 1-100 first
        self.sort_dub_files()

    def browse_dub_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if files:
            self.add_dub_files(files)

    def browse_dub_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Video Folder", "")
        if dir_path:
            self.add_dub_files([dir_path])

    def remove_dub_file(self):
        selected = self.list_dub_files.selectedItems()
        for item in selected:
            self.list_dub_files.takeItem(self.list_dub_files.row(item))
        self.save_dubber_session()

    def clear_dub_files(self):
        self.list_dub_files.clear()
        self.save_dubber_session()

    def sort_dub_files(self):
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        items_data = []
        for i in range(self.list_dub_files.count()):
            item = self.list_dub_files.item(i)
            items_data.append({
                "text": item.text(),
                "tooltip": item.toolTip(),
                "base_dir": item.data(Qt.ItemDataRole.UserRole)
            })
        
        # Sort items using natural sort key on the filename (item text)
        items_data.sort(key=lambda x: natural_sort_key(x["text"]))
        
        self.list_dub_files.clear()
        for data in items_data:
            item = QListWidgetItem(data["text"])
            item.setToolTip(data["tooltip"])
            if data["base_dir"]:
                item.setData(Qt.ItemDataRole.UserRole, data["base_dir"])
            self.list_dub_files.addItem(item)
        self.save_dubber_session()

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.txt_output_dir.text())
        if dir_path:
            self.txt_output_dir.setText(os.path.abspath(dir_path))

    # ----------------- QUEUE PIPELINE EXECUTION -----------------
    def update_voice_dropdown(self):
        self.combo_voice.blockSignals(True)
        self.combo_voice.clear()
        lang_code = self.combo_target_lang.currentData()
        voices = VOICES_MAP.get(lang_code, [])
        for name, value in voices:
            self.combo_voice.addItem(name, value)
            
        # Add basic VoxCPM options if they are not already in the VOICES_MAP list for this language
        has_voxcpm = any("voxcpm" in str(val) for _, val in voices)
        if not has_voxcpm:
            self.combo_voice.addItem("Auto VoxCPM Male/Female", "voxcpm-auto")
            self.combo_voice.addItem("Custom Voice Cloning", "voxcpm-custom")
            self.combo_voice.addItem("VoxCPM Female AI", "voxcpm-female")
            self.combo_voice.addItem("VoxCPM Male AI", "voxcpm-male")

        try:
            from settings.manager import load_settings
            settings = load_settings()
            saved_voices = settings.get("custom_cloned_voices", [])
            
            # Add the Auto Cloned option if saved voices exist
            if saved_voices:
                self.combo_voice.addItem("VoxCPM Auto Cloned Male/Female", "voxcpm-auto-cloned")
                
            for v in saved_voices:
                name_label = f"VoxCPM Clone: {v['name']}"
                value_val = f"voxcpm-custom|{v['path']}"
                self.combo_voice.addItem(name_label, value_val)
        except Exception as e:
            logger.error(f"Failed to load custom voices: {e}")
                
        # Set default/saved voice selection
        try:
            from settings.manager import load_settings
            settings = load_settings()
            saved_voice = settings.get("target_voice", "")
            
            idx = -1
            if saved_voice:
                idx = self.combo_voice.findData(saved_voice)
                
            # If no saved voice found, default to Piseth for Khmer
            if idx < 0 and lang_code == "km":
                idx = self.combo_voice.findData("km-KH-PisethNeural")
                
            if idx >= 0:
                self.combo_voice.setCurrentIndex(idx)
        except Exception as e:
            logger.error(f"Failed to apply default voice selection: {e}")
            
        self.combo_voice.blockSignals(False)
        self.on_voice_changed()

    def on_voice_changed(self):
        val = self.combo_voice.currentData()
        is_custom = (val == "voxcpm-custom")
        self.lbl_custom_voice.setVisible(is_custom)
        self.custom_voice_widget.setVisible(is_custom)
        
        # Enable delete button only if it's a saved cloned voice (starts with voxcpm-custom| )
        is_deletable = bool(val and isinstance(val, str) and val.startswith("voxcpm-custom|"))
        self.btn_delete_cloned_voice.setEnabled(is_deletable)
        
        # Save selected voice to settings
        if val is not None:
            try:
                from settings.manager import load_settings, save_settings
                settings = load_settings()
                settings["target_voice"] = val
                save_settings(settings)
            except Exception as e:
                logger.error(f"Failed to save target_voice: {e}")

    def browse_custom_voice_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Sample", "", "Audio Files (*.wav *.mp3 *.m4a *.ogg);;All Files (*)"
        )
        if file_path:
            self.txt_custom_voice_path.setText(os.path.abspath(file_path))

    def save_custom_voice_path(self, text):
        path = text.strip()
        if path and os.path.exists(path):
            try:
                from settings.manager import load_settings, save_settings
                settings = load_settings()
                settings["custom_voice_path"] = path
                save_settings(settings)
            except Exception as e:
                logger.error(f"Failed to save custom_voice_path: {e}")

    def save_gemini_api_key(self, text):
        key = text.strip()
        try:
            from settings.manager import load_settings, save_settings
            settings = load_settings()
            settings["gemini_api_key"] = key
            save_settings(settings)
        except Exception as e:
            logger.error(f"Failed to save gemini_api_key from dubber UI: {e}")

    def on_translation_engine_changed(self):
        engine = self.combo_trans_engine.currentData()
        is_gemini = (engine == "gemini")
        self.lbl_gemini_model.setVisible(is_gemini)
        self.combo_gemini_model.setVisible(is_gemini)
        self.lbl_gemini_key.setVisible(is_gemini)
        self.container_gemini.setVisible(is_gemini)
        
        # Save to settings
        try:
            from settings.manager import load_settings, save_settings
            settings = load_settings()
            settings["translation_engine"] = engine
            save_settings(settings)
        except Exception as e:
            logger.error(f"Failed to save translation_engine: {e}")

    def on_gemini_model_changed(self):
        model_name = self.combo_gemini_model.currentData()
        try:
            from settings.manager import load_settings, save_settings
            settings = load_settings()
            settings["gemini_model"] = model_name
            save_settings(settings)
        except Exception as e:
            logger.error(f"Failed to save gemini_model: {e}")

    def show_gemini_key_manager(self):
        dialog = GeminiKeyManagerDialog(self)
        dialog.exec()

    def verify_gemini_key(self):
        key = self.txt_gemini_api_key.text().strip()
        if not key:
            QMessageBox.warning(self, "Validation Error", "Please enter a Gemini API Key first.")
            return
            
        self.btn_check_gemini.setEnabled(False)
        self.btn_check_gemini.setText("Testing...")
        
        self.key_verifier = GeminiKeyVerifier(key)
        self.key_verifier.finished_signal.connect(self.on_gemini_verification_finished)
        self.key_verifier.start()
        
    def on_gemini_verification_finished(self, success, message):
        self.btn_check_gemini.setEnabled(True)
        self.btn_check_gemini.setText("Verify")
        if success:
            QMessageBox.information(self, "API Key Verified", message)
        else:
            QMessageBox.critical(self, "API Key Error", message)

    def save_custom_voice_profile(self):
        path = self.txt_custom_voice_path.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Validation Error", "Please select a valid voice recording file first.")
            return
            
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Custom Voice", "Enter a name for this cloned voice:")
        if ok and name.strip():
            voice_name = name.strip()
            try:
                from settings.manager import load_settings, save_settings
                settings = load_settings()
                custom_voices = settings.get("custom_cloned_voices", [])
                
                # Check if name already exists
                existing = [v for v in custom_voices if v["name"].lower() == voice_name.lower()]
                if existing:
                    existing[0]["path"] = path
                else:
                    custom_voices.append({"name": voice_name, "path": path})
                    
                settings["custom_cloned_voices"] = custom_voices
                save_settings(settings)
                
                QMessageBox.information(self, "Success", f"Voice '{voice_name}' saved successfully!")
                
                self.update_voice_dropdown()
                
                # Select the newly saved voice in the dropdown
                # Select the newly saved voice in the dropdown
                target_val = f"voxcpm-custom|{path}"
                for index in range(self.combo_voice.count()):
                    if self.combo_voice.itemData(index) == target_val:
                        self.combo_voice.setCurrentIndex(index)
                        break
            except Exception as e:
                logger.error(f"Failed to save custom voice profile: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save voice profile: {e}")

    def delete_custom_voice_profile(self):
        val = self.combo_voice.currentData()
        if not val or not isinstance(val, str) or not val.startswith("voxcpm-custom|"):
            return
            
        path_to_delete = val.split("|", 1)[1]
        
        confirm = QMessageBox.question(
            self, "Confirm Deletion",
            "Are you sure you want to delete this custom cloned voice profile?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                from settings.manager import load_settings, save_settings
                settings = load_settings()
                custom_voices = settings.get("custom_cloned_voices", [])
                filtered_voices = [v for v in custom_voices if v["path"] != path_to_delete]
                settings["custom_cloned_voices"] = filtered_voices
                save_settings(settings)
                self.update_voice_dropdown()
                QMessageBox.information(self, "Success", "Custom cloned voice profile removed successfully!")
            except Exception as e:
                logger.error(f"Failed to delete custom cloned voice: {e}")
                QMessageBox.critical(self, "Error", f"Failed to delete voice: {e}")

    def edit_dub_interactive(self):
        if self.list_dub_files.count() == 0:
            QMessageBox.warning(self, "Validation Error", "Please add at least one video to the list.")
            return
            
        selected_items = self.list_dub_files.selectedItems()
        if not selected_items:
            item = self.list_dub_files.item(0)
        else:
            item = selected_items[0]
            
        video_path = item.toolTip()
        
        if self.combo_voice.currentData() == "voxcpm-custom":
            custom_path = self.txt_custom_voice_path.text().strip()
            if not custom_path or not os.path.exists(custom_path):
                QMessageBox.warning(self, "Validation Error", "Please select a valid custom voice recording file.")
                return
                
        self.edit_dub_for_video(video_path)

    def on_transcription_ready(self, video_path, segments_list, translated_texts, params):
        # Restore buttons state
        self.btn_edit_dub.setEnabled(True)
        self.btn_dub_now.setEnabled(True)
        self.btn_queue_job.setEnabled(True)
        
        # Open the editor dialog!
        dialog = DubberEditorDialog(video_path, segments_list, translated_texts, params, parent=self)
        dialog.exec()

    def on_interactive_progress(self, val):
        self.lbl_queue_status.setText(f"Transcribing {val}%")

    def on_interactive_status(self, msg):
        self.lbl_queue_status.setText(msg)

    def on_interactive_log(self, msg):
        self.log_area.append(msg)

    def on_interactive_finished(self, success, path_or_err, details):
        self.btn_edit_dub.setEnabled(True)
        self.btn_dub_now.setEnabled(True)
        self.btn_queue_job.setEnabled(True)
        self.lbl_queue_status.setText("Idle")
        if not success:
            QMessageBox.critical(self, "Error", f"Transcription failed:\n{path_or_err}")
        if hasattr(self, "interactive_worker"):
            try:
                self.interactive_worker.deleteLater()
            except Exception:
                pass
            del self.interactive_worker

    def queue_dubbing_job(self):
        if self.list_dub_files.count() == 0:
            QMessageBox.warning(self, "Validation Error", "Please add at least one video to the list.")
            return

        if self.combo_voice.currentData() == "voxcpm-custom":
            custom_path = self.txt_custom_voice_path.text().strip()
            if not custom_path or not os.path.exists(custom_path):
                QMessageBox.warning(self, "Validation Error", "Please select a valid custom voice recording file.")
                return

        export_dir = self.txt_output_dir.text()
        os.makedirs(export_dir, exist_ok=True)

        for i in range(self.list_dub_files.count()):
            item = self.list_dub_files.item(i)
            video_path = item.toolTip()
            base_dir = item.data(Qt.ItemDataRole.UserRole)

            job_output_dir = export_dir
            if base_dir:
                try:
                    rel_file_path = os.path.relpath(video_path, base_dir)
                    rel_dir_path = os.path.dirname(rel_file_path)
                    if rel_dir_path:
                        job_output_dir = os.path.join(export_dir, rel_dir_path)
                except Exception as e:
                    logger.error(f"Failed to calculate relative output dir: {e}")

            os.makedirs(job_output_dir, exist_ok=True)

            params = {
                "video_path": video_path,
                "src_lang": self.combo_source_lang.currentData(),
                "tgt_lang": self.combo_target_lang.currentData(),
                "voice": self.combo_voice.currentData(),
                "custom_voice_path": self.txt_custom_voice_path.text().strip() if self.combo_voice.currentData() == "voxcpm-custom" else None,
                "model_size": self.combo_model_size.currentText(),
                "device": self.combo_device.currentText(),
                "vol_original": self.combo_original_vol.currentData(),
                "vol_dubbed": self.combo_dubbed_vol.currentData(),
                "auto_speed": self.chk_auto_speed.isChecked(),
                "mute_vocals": self.chk_mute_vocals.isChecked(),
                "mute_thoughts": False,
                "match_echo": True,
                "telephone_effect": True,
                "output_dir": job_output_dir
            }

            job_id = str(uuid.uuid4())
            voice_short = params["voice"].split("-")[-1].replace("Neural", "") if "auto" not in params["voice"] else "auto"
            job = {
                "id": job_id,
                "type": "Dub",
                "name": os.path.basename(video_path),
                "inputs": [video_path],
                "output": os.path.join(job_output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_dubbed_{params['tgt_lang']}.mp4"),
                "details": f"Dub: {params['tgt_lang'].upper()} ({voice_short})",
                "status": "Pending",
                "progress": 0,
                "dubbing_params": params,
                "error_msg": None
            }
            self.jobs.append(job)

        self.refresh_table()
        self.update_overall_progress()
        self.save_dubber_session()

        if self.running:
            self.process_next_jobs()

    def dub_all_now(self):
        # Re-route: Queue the current files, and start the queue immediately
        if self.list_dub_files.count() == 0:
            QMessageBox.warning(self, "Validation Error", "Please add at least one video to the list.")
            return
        self.queue_dubbing_job()
        self.start_queue()

    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for idx, job in enumerate(self.jobs):
            self.table.insertRow(idx)
            
            # Name
            self.table.setItem(idx, 0, QTableWidgetItem(job["name"]))
            # Type
            self.table.setItem(idx, 1, QTableWidgetItem(job["type"]))
            # Source Media
            src_str = os.path.basename(job["inputs"][0])
            self.table.setItem(idx, 2, QTableWidgetItem(src_str))
            # Operation Details
            self.table.setItem(idx, 3, QTableWidgetItem(job["details"]))
            
            # Status
            status_item = QTableWidgetItem(job["status"])
            if job["status"] == "Completed":
                status_item.setForeground(QColor("#10b981"))
            elif job["status"] == "Failed":
                status_item.setForeground(QColor("#ef4444"))
                if job.get("error_msg"):
                    status_item.setToolTip(job["error_msg"])
            elif job["status"] == "Processing":
                status_item.setForeground(QColor("#3b82f6"))
            elif job["status"] == "Cancelled" or job["status"] == "Paused":
                status_item.setForeground(QColor("#718096"))
            self.table.setItem(idx, 4, status_item)
            
            # Progress
            self.table.setItem(idx, 5, QTableWidgetItem(f"{job['progress']}%"))
            
        self.table.blockSignals(False)

    def start_queue(self):
        if not self.jobs:
            QMessageBox.information(self, "Empty Queue", "No queued dubbing jobs in the list.")
            return

        has_pending = any(j["status"] == "Pending" for j in self.jobs)
        if not has_pending:
            reply = QMessageBox.question(
                self, "Reset Queue", "All jobs are finished. Reset completed/failed tasks to retry?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                for job in self.jobs:
                    job["status"] = "Pending"
                    job["progress"] = 0
                    job["error_msg"] = None
                self.refresh_table()
            else:
                return

        self.running = True
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_dub_now.setEnabled(False)
        self.btn_edit_dub.setEnabled(False)
        self.btn_queue_job.setEnabled(False)
        self.lbl_queue_status.setText("Processing")
        self.lbl_queue_status.setStyleSheet("color: #3b82f6; border-color: #3b82f6; background-color: #1a1c26;")
        
        self.process_next_jobs()

    def pause_queue(self):
        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_dub_now.setEnabled(True)
        self.btn_edit_dub.setEnabled(True)
        self.btn_queue_job.setEnabled(True)
        self.lbl_queue_status.setText("Paused")
        self.lbl_queue_status.setStyleSheet("color: #718096; border-color: #718096; background-color: #1a1c26;")

        # Terminate active threads asynchronously without freezing the GUI
        for j_id, worker in list(self.active_workers.items()):
            try:
                worker.cancel()
            except Exception:
                pass
            self.finished_workers.append(worker)
            
        for job in self.jobs:
            if job["status"] == "Processing":
                job["status"] = "Pending"
                job["progress"] = 0

        self.active_workers.clear()
        self.finished_workers = [w for w in self.finished_workers if w.isRunning()]
        self.refresh_table()
        self.update_overall_progress()
        clear_whisper_model_cache()

    def process_next_jobs(self):
        if not self.running:
            return

        num_active = len(self.active_workers)
        slots = self.max_parallel - num_active

        if slots <= 0:
            return

        launched_any = False
        for job in self.jobs:
            if slots <= 0:
                break

            if job["status"] == "Pending" and job["id"] not in self.active_workers:
                job["status"] = "Processing"
                job["progress"] = 0
                launched_any = True

                worker = DubberWorker(job["dubbing_params"])
                self.active_workers[job["id"]] = worker

                # Bind signals
                worker.progress.connect(lambda val, j_id=job["id"]: self.on_job_progress(j_id, val))
                worker.status.connect(lambda msg, j_id=job["id"]: self.on_job_status_msg(j_id, msg))
                worker.log.connect(lambda msg, j_id=job["id"]: self.on_job_log_msg(j_id, msg))
                worker.finished.connect(lambda success, path_or_err, details, j_id=job["id"]: self.on_job_finished(j_id, success, path_or_err, details))
                
                worker.start()
                slots -= 1

        if launched_any:
            self.refresh_table()

        if len(self.active_workers) == 0:
            has_pending = any(j["status"] == "Pending" for j in self.jobs)
            if not has_pending:
                self.running = False
                self.btn_start.setEnabled(True)
                self.btn_pause.setEnabled(False)
                self.btn_dub_now.setEnabled(True)
                self.btn_edit_dub.setEnabled(True)
                self.btn_queue_job.setEnabled(True)
                self.lbl_queue_status.setText("Finished")
                self.lbl_queue_status.setStyleSheet("color: #10b981; border-color: #10b981; background-color: #1a1c26;")
                clear_whisper_model_cache()

        self.update_overall_progress()

    def on_job_progress(self, job_id: str, val: int):
        for job in self.jobs:
            if job["id"] == job_id:
                job["progress"] = val
                break
        # Update progress column cell
        for idx, job in enumerate(self.jobs):
            if job["id"] == job_id:
                item = self.table.item(idx, 5)
                if item:
                    item.setText(f"{val}%")
                break
        self.update_overall_progress()

    def on_job_status_msg(self, job_id: str, msg: str):
        filename = ""
        for job in self.jobs:
            if job["id"] == job_id:
                filename = job["name"]
                break
        if self.dashboard and hasattr(self.dashboard, "lbl_status"):
            self.dashboard.lbl_status.setText(f"[{filename}] {msg}")
        self.lbl_queue_status.setText(f"[{filename}] {msg}")

    def on_job_log_msg(self, job_id: str, msg: str):
        filename = ""
        for job in self.jobs:
            if job["id"] == job_id:
                filename = job["name"]
                break
        self.log_area.append(f"[{filename}] {msg}")

    def on_job_finished(self, job_id: str, success: bool, output_path: str, details: str):
        for job in self.jobs:
            if job["id"] == job_id:
                job["status"] = "Completed" if success else "Failed"
                job["progress"] = 100 if success else 0
                if not success:
                    job["error_msg"] = details
                else:
                    job["error_msg"] = None
                    job["output_file"] = output_path
                break

        worker = self.active_workers.pop(job_id, None)
        if worker:
            self.finished_workers.append(worker)
        self.finished_workers = [w for w in self.finished_workers if w.isRunning()]
        self.refresh_table()
        self.save_dubber_session()
        
        if self.dashboard and hasattr(self.dashboard, "refresh_library"):
            self.dashboard.refresh_library()
            
        self.process_next_jobs()

    def update_overall_progress(self):
        if not self.jobs:
            self.lbl_overall.setText("Overall progress: 0 / 0 jobs completed")
            self.progress_overall.setValue(0)
            return

        completed = sum(1 for j in self.jobs if j["status"] in ["Completed", "Failed", "Cancelled"])
        total = len(self.jobs)
        self.lbl_overall.setText(f"Overall progress: {completed} / {total} jobs completed")
        
        sum_prog = sum(j["progress"] for j in self.jobs)
        avg_prog = int(sum_prog / total)
        self.progress_overall.setValue(avg_prog)

    def on_max_parallel_changed(self, val: int):
        self.max_parallel = val
        if self.running:
            self.process_next_jobs()

    def remove_selected(self):
        selected_ranges = self.table.selectedRanges()
        rows = set()
        for r in selected_ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                rows.add(row)
        rows = sorted(list(rows))
        
        if not rows:
            return

        reply = QMessageBox.question(
            self, "Remove Tasks", f"Are you sure you want to remove the {len(rows)} selected task(s) from the queue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            for row in sorted(rows, reverse=True):
                if row < len(self.jobs):
                    job = self.jobs[row]
                    if job["id"] in self.active_workers:
                        worker = self.active_workers.pop(job["id"], None)
                        if worker:
                            try:
                                worker.cancel()
                            except Exception:
                                pass
                            self.finished_workers.append(worker)
                    self.jobs.pop(row)
            self.finished_workers = [w for w in self.finished_workers if w.isRunning()]
            self.refresh_table()
            self.update_overall_progress()
            self.save_dubber_session()
            if self.running:
                self.process_next_jobs()

    def clear_completed(self):
        row_idx = len(self.jobs) - 1
        removed = False
        while row_idx >= 0:
            if self.jobs[row_idx]["status"] in ["Completed", "Failed", "Cancelled"]:
                self.jobs.pop(row_idx)
                removed = True
            row_idx -= 1
        if removed:
            self.refresh_table()
            self.update_overall_progress()
            self.save_dubber_session()

    def show_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if not item:
            return
        row = self.table.row(item)
        if row < 0 or row >= len(self.jobs):
            return
            
        job = self.jobs[row]
        selected_rows = set()
        for r in self.table.selectedRanges():
            for r_idx in range(r.topRow(), r.bottomRow() + 1):
                selected_rows.add(r_idx)
        if row not in selected_rows:
            self.table.selectRow(row)

        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #16171f;
                border: 1px solid #2d3142;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu::item {
                padding: 6px 28px 6px 24px;
                color: #e2e8f0;
                font-size: 13px;
                border-radius: 4px;
                margin: 2px 0px;
            }
            QMenu::item:selected {
                background-color: #0084ff;
                color: #ffffff;
            }
            QMenu::item:disabled {
                color: #4a5568;
            }
        """)

        act_open_folder = QAction("📁 Open Output Folder", self)
        act_open_folder.setEnabled(job["status"] == "Completed")
        act_open_folder.triggered.connect(lambda: self.open_job_folder(job))

        act_edit_dub = QAction("📝 Edit Dub (Interactive)", self)
        video_path = job.get("inputs", [None])[0]
        act_edit_dub.setEnabled(job["type"] == "Dub" and video_path is not None and os.path.exists(video_path))
        act_edit_dub.triggered.connect(lambda: self.edit_dub_for_video(video_path, job.get("dubbing_params")))

        act_remove = QAction("🗑️ Remove Task", self)
        act_remove.triggered.connect(self.remove_selected)

        menu.addAction(act_open_folder)
        menu.addAction(act_edit_dub)
        menu.addAction(act_remove)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def show_list_context_menu(self, pos):
        item = self.list_dub_files.itemAt(pos)
        if not item:
            return
            
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #16171f;
                border: 1px solid #2d3142;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu::item {
                padding: 6px 28px 6px 24px;
                color: #e2e8f0;
                font-size: 13px;
                border-radius: 4px;
                margin: 2px 0px;
            }
            QMenu::item:selected {
                background-color: #0084ff;
                color: #ffffff;
            }
            QMenu::item:disabled {
                color: #4a5568;
            }
        """)
        
        act_edit = QAction("📝 Edit Dub (Interactive)", self)
        act_edit.triggered.connect(self.edit_dub_interactive)
        
        act_remove = QAction("🗑️ Remove from List", self)
        act_remove.triggered.connect(self.remove_dub_file)
        
        menu.addAction(act_edit)
        menu.addAction(act_remove)
        menu.exec(self.list_dub_files.viewport().mapToGlobal(pos))

    def edit_dub_for_video(self, video_path, dubbing_params=None):
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "Error", "Selected video file does not exist.")
            return
            
        export_dir = self.txt_output_dir.text()
        os.makedirs(export_dir, exist_ok=True)
        
        if dubbing_params:
            params = dubbing_params.copy()
            params["interactive"] = True
        else:
            params = {
                "video_path": video_path,
                "src_lang": self.combo_source_lang.currentData(),
                "tgt_lang": self.combo_target_lang.currentData(),
                "voice": self.combo_voice.currentData(),
                "custom_voice_path": self.txt_custom_voice_path.text().strip() if self.combo_voice.currentData() == "voxcpm-custom" else None,
                "model_size": self.combo_model_size.currentText(),
                "device": self.combo_device.currentText(),
                "vol_original": self.combo_original_vol.currentData(),
                "vol_dubbed": self.combo_dubbed_vol.currentData(),
                "auto_speed": self.chk_auto_speed.isChecked(),
                "mute_vocals": self.chk_mute_vocals.isChecked(),
                "mute_thoughts": False,
                "match_echo": True,
                "telephone_effect": True,
                "output_dir": export_dir,
                "interactive": True
            }
            
        self.log_area.clear()
        self.log_area.append(f"⏳ Starting interactive speech transcription for: {os.path.basename(video_path)}...")
        self.lbl_queue_status.setText("Transcribing...")
        
        self.btn_edit_dub.setEnabled(False)
        self.btn_dub_now.setEnabled(False)
        self.btn_queue_job.setEnabled(False)
        
        self.interactive_worker = DubberWorker(params)
        self.interactive_worker.transcription_ready.connect(self.on_transcription_ready)
        self.interactive_worker.progress.connect(self.on_interactive_progress)
        self.interactive_worker.status.connect(self.on_interactive_status)
        self.interactive_worker.log.connect(self.on_interactive_log)
        self.interactive_worker.finished.connect(self.on_interactive_finished)
        self.interactive_worker.start()

    def open_job_folder(self, job):
        output_path = job.get("output")
        if not output_path:
            return
        folder = os.path.dirname(output_path)
        os.makedirs(folder, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(folder)
            else:
                import subprocess
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, folder])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open output folder:\n{str(e)}")

    def get_local_model_path(self, model_size):
        from settings.manager import program_path
        return os.path.abspath(os.path.join(program_path, "models", "whisper", model_size))

    def is_model_downloaded(self, model_size):
        path = self.get_local_model_path(model_size)
        return is_whisper_model_valid(path, model_size)

    def update_model_status(self):
        model_size = self.combo_model_size.currentText()
        if self.is_model_downloaded(model_size):
            self.btn_download_model.setText("Downloaded (Local)")
            self.btn_download_model.setEnabled(False)
            self.btn_download_model.setStyleSheet("""
                QPushButton {
                    background-color: #1a3a2a;
                    color: #48bb78;
                    border: 1px solid #48bb78;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    font-size: 11px;
                }
            """)
        else:
            self.btn_download_model.setText("Download Offline")
            self.btn_download_model.setEnabled(True)
            self.btn_download_model.setStyleSheet("""
                QPushButton {
                    background-color: #1e202f;
                    color: #00a2ff;
                    border: 1px solid #00a2ff;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #00a2ff;
                    color: white;
                }
            """)

    def start_model_download(self):
        model_size = self.combo_model_size.currentText()
        output_dir = self.get_local_model_path(model_size)
        
        self.btn_download_model.setEnabled(False)
        self.btn_download_model.setText("Downloading...")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_queue_status.setText(f"Downloading Whisper '{model_size}' model...")
        
        self.log_area.append(f"\n==========================================")
        self.log_area.append(f"🤖 STARTING WHISPER '{model_size.upper()}' MODEL DOWNLOAD")
        self.log_area.append(f"==========================================")
        
        self.download_worker = WhisperDownloadWorker(model_size, output_dir)
        self.download_worker.progress.connect(self.log_area.append)
        self.download_worker.progress_val.connect(self.progress_overall.setValue)
        self.download_worker.finished.connect(self.handle_model_download_finished)
        self.download_worker.start()

    def handle_model_download_finished(self, success, details):
        self.update_model_status()
        self.progress_bar.setVisible(False)
        self.lbl_queue_status.setText("Ready")
        
        if success:
            QMessageBox.information(self, "Download Complete", details)
            self.log_area.append(f"✨ {details}\n")
        else:
            QMessageBox.critical(self, "Download Failed", f"Failed to download Whisper model:\n{details}")
            self.log_area.append(f"❌ Error: {details}\n")

    def load_dubber_session(self):
        try:
            from settings.manager import load_settings
            settings = load_settings()
            
            # Restore files list
            files = settings.get("dubber_files_list", [])
            self.list_dub_files.clear()
            for f_path in files:
                if f_path and os.path.exists(f_path):
                    from PyQt6.QtWidgets import QListWidgetItem
                    item = QListWidgetItem(os.path.basename(f_path))
                    item.setToolTip(f_path)
                    self.list_dub_files.addItem(item)
                    
            # Restore jobs queue
            saved_jobs = settings.get("dubber_jobs_queue", [])
            self.jobs = []
            for job in saved_jobs:
                if job.get("status") in ("Processing", "Verifying"):
                    job["status"] = "Pending"
                    job["progress"] = 0
                self.jobs.append(job)
            self.refresh_table()
            self.update_overall_progress()

            # Restore original video volume
            vol_orig = settings.get("dubber_vol_original")
            if vol_orig is not None:
                idx = self.combo_original_vol.findData(vol_orig)
                if idx >= 0:
                    self.combo_original_vol.setCurrentIndex(idx)
                    
            # Restore dubbed video volume
            vol_dub = settings.get("dubber_vol_dubbed")
            if vol_dub is not None:
                idx = self.combo_dubbed_vol.findData(vol_dub)
                if idx >= 0:
                    self.combo_dubbed_vol.setCurrentIndex(idx)

            # Restore model size and device
            self.combo_model_size.blockSignals(True)
            self.combo_device.blockSignals(True)
            try:
                model_size = settings.get("dubber_model_size")
                if model_size:
                    self.combo_model_size.setCurrentText(model_size)
                
                device = settings.get("dubber_device")
                if device:
                    self.combo_device.setCurrentText(device)
            finally:
                self.combo_model_size.blockSignals(False)
                self.combo_device.blockSignals(False)
                
            self.update_model_status()
        except Exception as e:
            logger.error(f"Failed to load dubber session: {e}")

    def save_dubber_session(self):
        try:
            from settings.manager import load_settings, save_settings
            settings = load_settings()
            
            # Save files list
            files = []
            for i in range(self.list_dub_files.count()):
                files.append(self.list_dub_files.item(i).toolTip())
            settings["dubber_files_list"] = files
            
            # Save jobs queue
            settings["dubber_jobs_queue"] = self.jobs

            # Save volumes
            settings["dubber_vol_original"] = self.combo_original_vol.currentData()
            settings["dubber_vol_dubbed"] = self.combo_dubbed_vol.currentData()

            # Save model size and device
            settings["dubber_model_size"] = self.combo_model_size.currentText()
            settings["dubber_device"] = self.combo_device.currentText()
            
            save_settings(settings)
        except Exception as e:
            logger.error(f"Failed to save dubber session: {e}")

    def check_and_prompt_resume(self):
        try:
            resume_data = load_resume_state()
            if not resume_data:
                return
                
            video_path = resume_data.get("video_path", "")
            if not video_path or not os.path.exists(video_path):
                clear_resume_state()
                return
                
            video_name = os.path.basename(video_path)
            
            confirm = QMessageBox.question(
                self, "Resume Interrupted Task",
                f"An interrupted dubbing job was found for:\n{video_name}\n\nDo you want to resume processing from where it was interrupted?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if confirm == QMessageBox.StandardButton.Yes:
                self.resume_job(resume_data)
            else:
                # User chose not to resume, clean up temp dir and clear state
                temp_dir = resume_data.get("temp_dir")
                if temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                clear_resume_state()
        except Exception as e:
            logger.error(f"Error checking resume state: {e}")

    def resume_job(self, resume_data):
        params = resume_data.get("params", {})
        is_interactive = params.get("interactive", False)
        
        self.log_area.clear()
        self.log_area.append("🔄 Resuming interrupted dubbing job...")
        self.lbl_queue_status.setText("Resuming...")
        
        self.btn_edit_dub.setEnabled(False)
        self.btn_dub_now.setEnabled(False)
        self.btn_queue_job.setEnabled(False)
        
        video_path = resume_data.get("video_path", "")
        # Add item to list if not present
        found = False
        for i in range(self.list_dub_files.count()):
            if self.list_dub_files.item(i).toolTip() == video_path:
                found = True
                break
        if not found:
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(os.path.basename(video_path))
            item.setToolTip(video_path)
            self.list_dub_files.addItem(item)
            
        if is_interactive:
            self.interactive_worker = DubberWorker(params)
            self.interactive_worker.resume_state = resume_data
            self.interactive_worker.transcription_ready.connect(self.on_transcription_ready)
            self.interactive_worker.progress.connect(self.on_interactive_progress)
            self.interactive_worker.status.connect(self.on_interactive_status)
            self.interactive_worker.log.connect(self.on_interactive_log)
            self.interactive_worker.finished.connect(self.on_interactive_finished)
            self.interactive_worker.start()
        else:
            # Look for existing job in self.jobs
            job_id = None
            for job in self.jobs:
                if job.get("inputs") and job["inputs"][0] == video_path:
                    job_id = job["id"]
                    job["status"] = "Processing"
                    break
            
            if not job_id:
                job_id = str(uuid.uuid4())
                job = {
                    "id": job_id,
                    "type": "Dub",
                    "name": os.path.basename(video_path),
                    "inputs": [video_path],
                    "output": os.path.join(params.get("output_dir", ""), f"{os.path.splitext(os.path.basename(video_path))[0]}_dubbed_{params.get('tgt_lang', 'km')}.mp4"),
                    "details": f"Dub: {params.get('tgt_lang', 'km').upper()}",
                    "status": "Processing",
                    "progress": 0,
                    "dubbing_params": params,
                    "error_msg": None
                }
                self.jobs.append(job)
            
            self.refresh_table()
            self.save_dubber_session()
            
            worker = DubberWorker(params)
            worker.resume_state = resume_data
            self.active_workers[job_id] = worker
            
            # Bind signals
            worker.progress.connect(lambda val, j_id=job_id: self.on_job_progress(j_id, val))
            worker.status.connect(lambda msg, j_id=job_id: self.on_job_status_msg(j_id, msg))
            worker.log.connect(lambda msg, j_id=job_id: self.on_job_log_msg(j_id, msg))
            worker.finished.connect(lambda success, path_or_err, details, j_id=job_id: self.on_job_finished(j_id, success, path_or_err, details))
            
            self.running = True
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.lbl_queue_status.setText("Processing...")
            self.lbl_queue_status.setStyleSheet("color: #3b82f6; border-color: #3b82f6; background-color: #1a1c26;")
            
            worker.start()

    def open_local_server_manager(self):
        dlg = VoxCPMServerManagerDialog(self)
        dlg.exec()

    def closeEvent(self, event):
        if hasattr(self, 'download_worker') and self.download_worker.isRunning():
            self.download_worker.cancel()
            self.download_worker.wait()
        if hasattr(self, 'active_workers') and self.active_workers:
            for worker in list(self.active_workers.values()):
                if worker.isRunning():
                    try:
                        worker.cancel()
                        worker.wait()
                    except Exception:
                        pass
        self.save_dubber_session()
        super().closeEvent(event)

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QLineEdit, QPushButton, QLabel, QMessageBox, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QThread

class GeminiKeyManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gemini API Keys Manager")
        self.setMinimumSize(600, 450)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QLabel {
                color: #cbd5e1;
            }
            QListWidget {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #475569;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #3b82f6;
                color: white;
            }
            QLineEdit, QTextEdit {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #3b82f6;
            }
            QPushButton {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #f8fafc;
            }
            QPushButton#btnAdd, QPushButton#btnSave {
                background-color: #3b82f6;
                color: white;
                border: none;
            }
            QPushButton#btnAdd:hover, QPushButton#btnSave:hover {
                background-color: #2563eb;
            }
            QPushButton#btnRemove {
                background-color: #ef4444;
                color: white;
                border: none;
            }
            QPushButton#btnRemove:hover {
                background-color: #dc2626;
            }
        """)

        self.verifier_threads = []
        self.init_ui()
        self.load_keys()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header Info
        header_label = QLabel("🔑 Manage Gemini API Keys")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f8fafc;")
        main_layout.addWidget(header_label)

        desc_label = QLabel("Add multiple API keys to balance translation workload and prevent rate limit (HTTP 429) errors. The application rotates through active keys round-robin.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 12px; color: #94a3b8; line-height: 1.4;")
        main_layout.addWidget(desc_label)

        # Center area layout: List on left, controls on right
        center_layout = QHBoxLayout()
        center_layout.setSpacing(14)

        # Left list widget
        self.list_keys = QListWidget()
        center_layout.addWidget(self.list_keys, 2)

        # Right control panel (Verify, Delete, etc.)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        right_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.btn_verify_selected = QPushButton("Verify Selected")
        self.btn_verify_selected.clicked.connect(self.verify_selected_key)
        right_panel.addWidget(self.btn_verify_selected)

        self.btn_verify_all = QPushButton("Verify All Keys")
        self.btn_verify_all.clicked.connect(self.verify_all_keys)
        right_panel.addWidget(self.btn_verify_all)

        right_panel.addWidget(QWidget()) # Spacer

        self.btn_remove_key = QPushButton("Remove Selected")
        self.btn_remove_key.setObjectName("btnRemove")
        self.btn_remove_key.clicked.connect(self.remove_selected_key)
        right_panel.addWidget(self.btn_remove_key)

        center_layout.addLayout(right_panel, 1)
        main_layout.addLayout(center_layout)

        # Add new key layout
        add_layout = QHBoxLayout()
        add_layout.setSpacing(10)

        self.txt_new_key = QTextEdit()
        self.txt_new_key.setPlaceholderText("Enter Gemini API Key(s) (separated by line, space, comma, or semicolon)...")
        self.txt_new_key.setFixedHeight(50)
        add_layout.addWidget(self.txt_new_key, 1)

        self.btn_add_key = QPushButton("Add Keys")
        self.btn_add_key.setObjectName("btnAdd")
        self.btn_add_key.setMinimumHeight(50)
        self.btn_add_key.clicked.connect(self.add_key)
        add_layout.addWidget(self.btn_add_key)

        main_layout.addLayout(add_layout)

        # Bottom buttons (Save & Cancel)
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        bottom_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save & Apply")
        self.btn_save.setObjectName("btnSave")
        self.btn_save.clicked.connect(self.save_and_close)
        bottom_layout.addWidget(self.btn_save)

        main_layout.addLayout(bottom_layout)

    def load_keys(self):
        from ui.video_dubber import get_gemini_keys
        keys = get_gemini_keys()
        for k in keys:
            self.add_key_to_list(k, "Untested", "#94a3b8")

    def add_key_to_list(self, raw_key, status, color_hex):
        masked_key = raw_key[:7] + "..." + raw_key[-4:] if len(raw_key) > 11 else "Invalid Length"
        item = QListWidgetItem()
        item.setText(f"🔑 {masked_key}  [{status}]")
        item.setData(Qt.ItemDataRole.UserRole, raw_key)
        item.setData(Qt.ItemDataRole.UserRole + 1, (status, color_hex))
        item.setForeground(Qt.GlobalColor.white)
        self.list_keys.addItem(item)
        self.update_item_appearance(item, status, color_hex)

    def update_item_appearance(self, item, status, color_hex):
        raw_key = item.data(Qt.ItemDataRole.UserRole)
        masked_key = raw_key[:7] + "..." + raw_key[-4:] if len(raw_key) > 11 else "Invalid Length"
        item.setText(f"🔑 {masked_key}  —  {status}")
        
        from PyQt6.QtGui import QColor, QBrush
        item.setForeground(QBrush(QColor("#f1f5f9")))
        item.setToolTip(f"Status: {status}\nFull Key: {raw_key[:10]}...")

    def add_key(self):
        input_text = self.txt_new_key.toPlainText().strip()
        if not input_text:
            QMessageBox.warning(self, "Validation Error", "Please enter one or more API Keys first.")
            return

        # Split by comma, semicolon, space, or newline
        import re
        keys = [k.strip() for k in re.split(r'[\s,;\n\r]+', input_text) if k.strip()]
        
        if not keys:
            QMessageBox.warning(self, "Validation Error", "No valid keys found in the input.")
            return

        added_count = 0
        duplicate_count = 0
        
        existing_keys = set()
        for i in range(self.list_keys.count()):
            existing_keys.add(self.list_keys.item(i).data(Qt.ItemDataRole.UserRole))

        for key in keys:
            if key in existing_keys:
                duplicate_count += 1
                continue
            self.add_key_to_list(key, "Untested", "#94a3b8")
            existing_keys.add(key)
            added_count += 1

        self.txt_new_key.clear()
        
        if added_count > 0:
            if duplicate_count > 0:
                QMessageBox.information(self, "Keys Added", f"Successfully added {added_count} key(s). {duplicate_count} duplicate key(s) ignored.")
        else:
            if duplicate_count > 0:
                QMessageBox.warning(self, "Duplicate Keys", "All entered keys are already in the list.")

    def remove_selected_key(self):
        selected_items = self.list_keys.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select a key to remove.")
            return
        for item in selected_items:
            self.list_keys.takeItem(self.list_keys.row(item))

    def verify_selected_key(self):
        selected_items = self.list_keys.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select a key to verify.")
            return
        item = selected_items[0]
        raw_key = item.data(Qt.ItemDataRole.UserRole)
        
        item.setData(Qt.ItemDataRole.UserRole + 1, ("Verifying...", "#eab308"))
        self.update_item_appearance(item, "🔍 Verifying...", "#eab308")
        
        from ui.video_dubber import GeminiKeyVerifier
        verifier = GeminiKeyVerifier(raw_key)
        verifier.finished_signal.connect(lambda success, msg, it=item: self.on_verification_finished(it, success, msg))
        verifier.start()
        self.verifier_threads.append(verifier)

    def verify_all_keys(self):
        if self.list_keys.count() == 0:
            QMessageBox.warning(self, "No Keys", "There are no keys to verify.")
            return
            
        for i in range(self.list_keys.count()):
            item = self.list_keys.item(i)
            raw_key = item.data(Qt.ItemDataRole.UserRole)
            item.setData(Qt.ItemDataRole.UserRole + 1, ("Verifying...", "#eab308"))
            self.update_item_appearance(item, "🔍 Verifying...", "#eab308")
            
            from ui.video_dubber import GeminiKeyVerifier
            verifier = GeminiKeyVerifier(raw_key)
            verifier.finished_signal.connect(lambda success, msg, it=item: self.on_verification_finished(it, success, msg))
            verifier.start()
            self.verifier_threads.append(verifier)

    def on_verification_finished(self, item, success, message):
        if success:
            status = "✅ Active"
            color = "#10b981"
        else:
            status = f"❌ Invalid ({message})"
            color = "#ef4444"
        item.setData(Qt.ItemDataRole.UserRole + 1, (status, color))
        self.update_item_appearance(item, status, color)

    def save_and_close(self):
        keys_list = []
        for i in range(self.list_keys.count()):
            item = self.list_keys.item(i)
            keys_list.append(item.data(Qt.ItemDataRole.UserRole))
            
        final_keys_str = ", ".join(keys_list)
        try:
            from settings.manager import load_settings, save_settings
            settings = load_settings()
            settings["gemini_api_key"] = final_keys_str
            save_settings(settings)
            
            if self.parent():
                if hasattr(self.parent(), 'txt_gemini_api_key'):
                    self.parent().txt_gemini_api_key.setText(final_keys_str)
                elif hasattr(self.parent(), 'txt_gemini_key'):
                    self.parent().txt_gemini_key.setText(final_keys_str)
            
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error Saving", f"Failed to save keys: {e}")

from PyQt6.QtWidgets import QDialog

class ServerStatusWorker(QThread):
    # Signals: running, loaded, error_msg, log_detail
    finished = pyqtSignal(bool, bool, str, str)

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def run(self):
        running = False
        loaded = False
        error_msg = ""
        log_detail = ""
        try:
            running, loaded, error_msg = self.manager.is_running()
        except Exception as e:
            error_msg = str(e)
            
        # Parse the last few lines of the local log to get detailed startup status
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_path = os.path.join(project_root, "local_voxcpm_server.log")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    
                valid_lines = [l.strip() for l in lines if l.strip()]
                if valid_lines:
                    # Filter out recurrent FastAPI health checks to avoid noise
                    clean_lines = [l for l in valid_lines if "GET /health" not in l]
                    if clean_lines:
                        # Extract the last 2 relevant lines for detail display
                        last_lines = clean_lines[-2:]
                        cleaned_output = []
                        for l in last_lines:
                            if "|" in l and "%" in l:
                                # Clean up tqdm progress bars for display
                                parts = l.split("|")
                                if len(parts) >= 3:
                                    pct = parts[0].strip()
                                    details = parts[-1].strip()
                                    cleaned_output.append(f"Warming up model: {pct} ({details})")
                                else:
                                    cleaned_output.append(l)
                            else:
                                cleaned_output.append(l)
                        log_detail = " | ".join(cleaned_output)
        except Exception as le:
            logger.debug(f"Failed to read local server log: {le}")
            
        self.finished.emit(running, loaded, error_msg or "", log_detail)

class VoxCPMServerManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VoxCPM Local Server Manager")
        self.setMinimumWidth(500)
        
        # Load settings and define values first
        from settings.manager import load_settings
        settings = load_settings()
        
        # Default settings
        default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "VoxCPM2")
        self.model_dir = settings.get("voxcpm_model_dir", default_dir)
        self.host = settings.get("voxcpm_host", "127.0.0.1")
        self.port = int(settings.get("voxcpm_port", 8000))
        self.device = get_safe_device(settings.get("voxcpm_device", "cuda"))
        self.cfg_value = float(settings.get("voxcpm_cfg_value", 2.0))
        self.inference_timesteps = int(settings.get("voxcpm_inference_timesteps", 10))
        self.load_denoiser = settings.get("voxcpm_load_denoiser", False)
        self.backend = settings.get("voxcpm_backend", "local")
        
        # Build UI (uses cfg_value, inference_timesteps, load_denoiser)
        self.init_ui()
        
        # Populate UI text fields
        self.txt_model_dir.setText(self.model_dir)
        self.txt_host.setText(self.host)
        self.txt_port.setText(str(self.port))
        self.combo_device.setCurrentText(self.device)
        
        # Connect settings changes to auto-save
        self.txt_model_dir.textChanged.connect(self.save_voxcpm_settings)
        self.txt_host.textChanged.connect(self.save_voxcpm_settings)
        self.txt_port.textChanged.connect(self.save_voxcpm_settings)
        self.combo_device.currentTextChanged.connect(self.save_voxcpm_settings)
        self.spin_cfg.valueChanged.connect(self.save_voxcpm_settings)
        self.spin_steps.valueChanged.connect(self.save_voxcpm_settings)
        self.chk_denoiser.stateChanged.connect(self.save_voxcpm_settings)
        
        # Initialize manager
        from app.voxcpm_manager import LocalVoxCPMServerManager
        self.manager = LocalVoxCPMServerManager()
        self.download_worker = None
        self.status_worker = None
        
        # Configure manager
        self.manager.configure(self.host, self.port, self.device, self.model_dir)
        
        # Timer for polling connection status
        from PyQt6.QtCore import QTimer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.check_server_status)
        self.status_timer.start(3000) # check status every 3 seconds
        self.check_server_status()

    def init_ui(self):
        from PyQt6.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
            QPushButton, QProgressBar, QComboBox, QMessageBox
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # --- Group 1: Model Downloader ---
        download_group = QGroupBox("📥 VoxCPM Model Downloader")
        dl_lay = QVBoxLayout(download_group)
        dl_lay.setSpacing(8)
        
        # Model path selection
        dir_lay = QHBoxLayout()
        dir_lay.addWidget(QLabel("Model Directory:"))
        self.txt_model_dir = QLineEdit()
        self.btn_browse_dir = QPushButton("📁")
        self.btn_browse_dir.setFixedWidth(30)
        self.btn_browse_dir.clicked.connect(self.browse_model_directory)
        dir_lay.addWidget(self.txt_model_dir)
        dir_lay.addWidget(self.btn_browse_dir)
        dl_lay.addLayout(dir_lay)
        
        # Download Controls
        ctrl_lay = QHBoxLayout()
        self.btn_download = QPushButton("⬇️ Download Model")
        self.btn_download.clicked.connect(self.start_download)
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_download)
        self.btn_resume = QPushButton("▶️ Resume")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self.resume_download)
        ctrl_lay.addWidget(self.btn_download)
        ctrl_lay.addWidget(self.btn_pause)
        ctrl_lay.addWidget(self.btn_resume)
        dl_lay.addLayout(ctrl_lay)
        
        # Progress Bar & Labels
        self.lbl_status = QLabel("Status: Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.lbl_details = QLabel("Speed: - MB/s | ETA: - | Completed: 0/9 files")
        
        dl_lay.addWidget(self.lbl_status)
        dl_lay.addWidget(self.progress_bar)
        dl_lay.addWidget(self.lbl_details)
        
        layout.addWidget(download_group)
        
        # --- Group 2: Server Management ---
        server_group = QGroupBox("⚙️ Local Server Controls")
        srv_lay = QVBoxLayout(server_group)
        srv_lay.setSpacing(8)
        
        # Host/Port Settings
        set_lay = QHBoxLayout()
        set_lay.addWidget(QLabel("Host:"))
        self.txt_host = QLineEdit()
        set_lay.addWidget(self.txt_host)
        set_lay.addWidget(QLabel("Port:"))
        self.txt_port = QLineEdit()
        self.txt_port.setFixedWidth(60)
        set_lay.addWidget(self.txt_port)
        set_lay.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "cpu"])
        set_lay.addWidget(self.combo_device)
        srv_lay.addLayout(set_lay)
        
        # Generation Parameters (CFG, Steps, Denoiser)
        from PyQt6.QtWidgets import QDoubleSpinBox, QSpinBox, QCheckBox
        param_lay = QHBoxLayout()
        param_lay.addWidget(QLabel("CFG Scale:"))
        self.spin_cfg = QDoubleSpinBox()
        self.spin_cfg.setRange(1.0, 10.0)
        self.spin_cfg.setSingleStep(0.5)
        self.spin_cfg.setValue(self.cfg_value)
        param_lay.addWidget(self.spin_cfg)
        
        param_lay.addWidget(QLabel("Inference Steps:"))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(2, 50)
        self.spin_steps.setValue(self.inference_timesteps)
        param_lay.addWidget(self.spin_steps)
        
        self.chk_denoiser = QCheckBox("Enable Denoiser")
        self.chk_denoiser.setChecked(self.load_denoiser)
        param_lay.addWidget(self.chk_denoiser)
        srv_lay.addLayout(param_lay)
        
        # Backend Settings Selection
        backend_lay = QHBoxLayout()
        backend_lay.addWidget(QLabel("VoxCPM Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["Local Server", "openbmb (HF Space)"])
        current_backend = "Local Server" if self.backend == "local" else "openbmb (HF Space)"
        self.combo_backend.setCurrentText(current_backend)
        self.combo_backend.currentTextChanged.connect(self.save_voxcpm_settings)
        backend_lay.addWidget(self.combo_backend)
        srv_lay.addLayout(backend_lay)
        
        # Server Action buttons
        act_lay = QHBoxLayout()
        self.btn_start_server = QPushButton("🟢 Start Server")
        self.btn_start_server.clicked.connect(self.start_server)
        self.btn_stop_server = QPushButton("🔴 Stop Server")
        self.btn_stop_server.clicked.connect(self.stop_server)
        self.btn_test_conn = QPushButton("⚡ Test Connection")
        self.btn_test_conn.clicked.connect(self.test_connection)
        act_lay.addWidget(self.btn_start_server)
        act_lay.addWidget(self.btn_stop_server)
        act_lay.addWidget(self.btn_test_conn)
        srv_lay.addLayout(act_lay)
        
        # Server Status display
        self.lbl_server_status = QLabel("Server Connection Status: Offline")
        self.lbl_server_status.setStyleSheet("color: #ef4444; font-weight: bold;")
        srv_lay.addWidget(self.lbl_server_status)
        
        self.lbl_server_detail = QLabel("")
        self.lbl_server_detail.setWordWrap(True)
        self.lbl_server_detail.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic; margin-top: 2px;")
        srv_lay.addWidget(self.lbl_server_detail)
        
        layout.addWidget(server_group)

    def browse_model_directory(self):
        from PyQt6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "Select Model Directory", self.model_dir)
        if dir_path:
            self.model_dir = os.path.abspath(dir_path)
            self.txt_model_dir.setText(self.model_dir)
            self.save_voxcpm_settings()
            
    def save_voxcpm_settings(self):
        from settings.manager import load_settings, save_settings
        try:
            settings = load_settings()
            settings["voxcpm_model_dir"] = self.txt_model_dir.text().strip()
            settings["voxcpm_host"] = self.txt_host.text().strip()
            settings["voxcpm_port"] = int(self.txt_port.text().strip())
            settings["voxcpm_device"] = self.combo_device.currentText()
            settings["voxcpm_cfg_value"] = float(self.spin_cfg.value())
            settings["voxcpm_inference_timesteps"] = int(self.spin_steps.value())
            settings["voxcpm_load_denoiser"] = self.chk_denoiser.isChecked()
            selected_backend = "local" if self.combo_backend.currentText() == "Local Server" else "openbmb"
            settings["voxcpm_backend"] = selected_backend
            save_settings(settings)
            
            self.model_dir = settings["voxcpm_model_dir"]
            self.host = settings["voxcpm_host"]
            self.port = settings["voxcpm_port"]
            self.device = get_safe_device(settings["voxcpm_device"])
            self.cfg_value = settings["voxcpm_cfg_value"]
            self.inference_timesteps = settings["voxcpm_inference_timesteps"]
            self.load_denoiser = settings["voxcpm_load_denoiser"]
            self.backend = settings["voxcpm_backend"]
            self.manager.configure(self.host, self.port, self.device, self.model_dir)
        except Exception:
            pass

    def start_download(self):
        self.save_voxcpm_settings()
        from app.voxcpm_manager import ModelDownloadWorker
        self.download_worker = ModelDownloadWorker(self.model_dir)
        self.download_worker.progress.connect(self.on_download_progress)
        self.download_worker.file_completed.connect(self.on_file_completed)
        self.download_worker.all_completed.connect(self.on_all_completed)
        self.download_worker.paused.connect(self.on_download_paused)
        self.download_worker.error.connect(self.on_download_error)
        self.download_worker.status_msg.connect(self.lbl_status.setText)
        
        self.download_completed_count = 0
        self.btn_download.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        
        self.download_worker.start()
        
    def pause_download(self):
        if self.download_worker:
            self.download_worker.is_paused = True
            
    def resume_download(self):
        self.start_download()
        
    def on_download_progress(self, file_name, bytes_downloaded, total_bytes, speed_mbps, eta):
        pct = int((bytes_downloaded / total_bytes) * 100) if total_bytes > 0 else 0
        self.progress_bar.setValue(pct)
        
        mb_dl = bytes_downloaded / (1024 * 1024)
        mb_tot = total_bytes / (1024 * 1024)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta)) if eta > 0 else "estimating..."
        self.lbl_details.setText(
            f"Downloading {file_name}: {mb_dl:.1f}MB / {mb_tot:.1f}MB | "
            f"Speed: {speed_mbps:.1f} MB/s | ETA: {eta_str} | Completed: {self.download_completed_count}/9 files"
        )
        
    def on_file_completed(self, file_name):
        self.download_completed_count += 1
        self.lbl_details.setText(f"Completed: {self.download_completed_count}/9 files")
        
    def on_all_completed(self):
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Status: Download completed successfully!")
        self.lbl_details.setText("All 9 model files are present and verified.")
        self.btn_download.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        QMessageBox.information(self, "Success", "VoxCPM Model downloaded and verified successfully!")
        
    def on_download_paused(self):
        self.lbl_status.setText("Status: Paused")
        self.btn_download.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        
    def on_download_error(self, error_msg):
        self.lbl_status.setText("Status: Error")
        self.lbl_details.setText(error_msg)
        self.btn_download.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        QMessageBox.critical(self, "Download Error", error_msg)

    def start_server(self):
        self.save_voxcpm_settings()
        
        from app.voxcpm_manager import VOXCPM_FILES
        missing = []
        for f in VOXCPM_FILES:
            fp = os.path.join(self.model_dir, f["name"])
            if not os.path.exists(fp) or os.path.getsize(fp) != f["size"]:
                missing.append(f["name"])
                
        if missing:
            confirm = QMessageBox.question(
                self, "Missing Files",
                f"The following model files are missing or incomplete:\n" + "\n".join(missing[:3]) + 
                (f"\n...and {len(missing)-3} more." if len(missing) > 3 else "") + 
                "\n\nDo you want to start the server anyway (it will download weights from Hugging Face on-the-fly)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
                
        self.lbl_server_status.setText("Server Connection Status: Launching...")
        self.lbl_server_status.setStyleSheet("color: #eab308; font-weight: bold;")
        
        success, msg = self.manager.start_server()
        if success:
            self.check_server_status()
        else:
            QMessageBox.critical(self, "Server Launch Error", msg)
            self.check_server_status()
            
    def stop_server(self):
        self.manager.stop_server()
        self.check_server_status()
        
    def test_connection(self):
        self.save_voxcpm_settings()
        self.check_server_status()
        running, loaded, error_msg = self.manager.is_running()
        if running:
            if error_msg:
                QMessageBox.critical(self, "Connection Test", f"Server is connected, but model loading failed with error:\n{error_msg}")
            elif loaded:
                QMessageBox.information(self, "Connection Test", "Connection successful! Local VoxCPM model is fully loaded and ready.")
            else:
                QMessageBox.warning(self, "Connection Test", "Server is running, but the VoxCPM model weights are still loading in the background. Please wait.")
        else:
            QMessageBox.critical(self, "Connection Test", "Failed to connect to local VoxCPM server. Make sure it is started.")
            
    def check_server_status(self):
        if self.status_worker and self.status_worker.isRunning():
            return
        self.status_worker = ServerStatusWorker(self.manager)
        self.status_worker.finished.connect(self.on_status_checked)
        self.status_worker.start()

    def on_status_checked(self, running, loaded, error_msg, log_detail):
        if not error_msg:
            error_msg = None

        if running:
            if error_msg:
                self.lbl_server_status.setText("Server Connection Status: ❌ Load Error (see log)")
                self.lbl_server_status.setToolTip(error_msg)
                self.lbl_server_status.setStyleSheet("color: #ef4444; font-weight: bold;")
                self.lbl_server_detail.setText(log_detail)
            elif loaded:
                self.lbl_server_status.setText("Server Connection Status: 🟢 Connected & Model Loaded")
                self.lbl_server_status.setToolTip("")
                self.lbl_server_status.setStyleSheet("color: #22c55e; font-weight: bold;")
                self.lbl_server_detail.setText(log_detail or "Ready for speech synthesis.")
            else:
                self.lbl_server_status.setText("Server Connection Status: 🟡 Connected (Loading weights...)")
                self.lbl_server_status.setToolTip("")
                self.lbl_server_status.setStyleSheet("color: #eab308; font-weight: bold;")
                self.lbl_server_detail.setText(log_detail or "Loading weights into memory...")
        else:
            self.lbl_server_status.setText("Server Connection Status: 🔴 Offline")
            self.lbl_server_status.setToolTip("")
            self.lbl_server_status.setStyleSheet("color: #ef4444; font-weight: bold;")
            self.lbl_server_detail.setText("")
            
    def closeEvent(self, event):
        self.status_timer.stop()
        if self.status_worker:
            self.status_worker.quit()
            self.status_worker.wait()
        if self.download_worker:
            self.download_worker.is_cancelled = True
        event.accept()
