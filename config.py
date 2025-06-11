# config.py - 설정 및 상수 관리

import os

# 🔥 기본 단어 세트들 정의
BASIC_VERBS = {
    # 감각/지각 동사
    "feel",
    "see",
    "hear",
    "smell",
    "taste",
    "touch",
    "look",
    "watch",
    "listen",
    # 상태 동사
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "seem",
    "appear",
    "become",
    "stay",
    "remain",
    # 기본 행동 동사
    "go",
    "come",
    "get",
    "give",
    "take",
    "make",
    "do",
    "have",
    "say",
    "tell",
    "speak",
    "eat",
    "drink",
    "sleep",
    "walk",
    "run",
    "sit",
    "stand",
    "lie",
    "live",
    "die",
    # 감정 동사
    "love",
    "like",
    "hate",
    "want",
    "need",
    "hope",
    "wish",
    "fear",
    "worry",
    "care",
    # 사고 동사
    "think",
    "know",
    "believe",
    "understand",
    "remember",
    "forget",
    "learn",
    "study",
    # 기본 작업 동사
    "work",
    "play",
    "help",
    "start",
    "stop",
    "finish",
    "open",
    "close",
    "turn",
    "move",
}

BASIC_ADJECTIVES = {
    "good",
    "bad",
    "big",
    "small",
    "old",
    "new",
    "young",
    "hot",
    "cold",
    "warm",
    "happy",
    "sad",
    "angry",
    "tired",
    "hungry",
    "thirsty",
    "easy",
    "hard",
    "difficult",
    "important",
    "interesting",
    "beautiful",
    "ugly",
    "clean",
    "dirty",
    "fast",
    "slow",
}

BASIC_NOUNS = {
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "people",
    "person",
    "family",
    "friend",
    "house",
    "home",
    "school",
    "work",
    "job",
    "money",
    "time",
    "day",
    "night",
    "year",
}

# 기본 설정값
DEFAULT_SETTINGS = {
    "DIFFICULTY_THRESHOLD": 3.0,
    "GPT_CALL_LIMIT": 1000,
    "USER_PRIORITY": 1,
    "CACHE_FILE": "gpt_cache.json",
    "MIN_WORD_LENGTH": 4,
    "EASY_WORDS_CACHE": "elementary_words.pkl",
    "MAX_TOKENS": 200000,
    "USE_CACHE": True,
    "USE_INTEGRATED_CONTEXTUAL": True,
    "USE_INTEGRATED_DIFFICULTY": True,
    "ENHANCED_MEANING_GENERATION": True,
    "USE_GPT_DIFFICULTY_FILTER": True,
}


# 파일 경로 설정
class FilePaths:
    CACHE_DIR = "cache"
    DATA_DIR = "data"
    USER_WORDS_FILE = "단어DB.csv"
    DEFAULT_INPUT = "지문DB.csv"
    DEFAULT_OUTPUT = "vocabulary_advanced.xlsx"

    @classmethod
    def ensure_directories(cls):
        """필요한 디렉토리 생성"""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)


# 품사 변환 매핑
POS_MAPPING = {"NOUN": "명사", "VERB": "동사", "ADJ": "형용사", "ADV": "부사"}

# 인코딩 시도 순서
ENCODING_ORDER = ["utf-8", "cp949", "euc-kr", "latin1", "utf-8-sig"]

# 텍스트 컬럼 후보들
TEXT_COLUMNS = ["content", "지문", "text", "텍스트", "본문", "Content", "TEXT"]

# 쉬운 단어 파일 후보들
EASY_WORDS_FILES = {
    "excel": ["easy_words.xlsx", "elementary_words.xlsx", "basic_words.xlsx"],
    "csv": ["easy_words.csv", "elementary_words.csv", "basic_words.csv"],
}


# 22컬럼 표준 구조 정의
VOCABULARY_COLUMNS = [
    "교재ID",
    "교재명",
    "지문ID",
    "순서",
    "지문",
    "단어",
    "원형",
    "품사",
    "뜻(한글)",
    "뜻(영어)",
    "동의어",
    "반의어",
    "문맥",
    "분리형여부",
    "신뢰도",
    "사용자DB매칭",
    "매칭방식",
    "패턴정보",
    "문맥적의미",
    "동의어신뢰도",
    "처리방식",
    "포함이유",
]

# 컬럼 타입 정의
COLUMN_TYPES = {
    "교재ID": "float64",
    "순서": "int64",
    "분리형여부": "bool",
    "신뢰도": "float64",
    "사용자DB매칭": "bool",
    "동의어신뢰도": "float64",
}
