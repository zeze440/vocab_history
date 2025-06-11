import os
import json
import time
import pickle
import logging
import openai
import re
import glob
import requests
import numpy as np
import pandas as pd
from safe_data_utils import (
    safe_get_column_value,
    safe_string_operation,
    safe_numeric_operation,
)
from synonym_antonym_module import SynonymAntonymExtractor
from missing_methods import MissingMethodsMixin

from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from typing import List, Dict, Any, Set, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import hashlib


# 🔥 3. 어근 중복 제거 전역 함수 추가 (파일 맨 위쪽에)
def enhanced_filter_synonyms_antonyms(candidates, original_word, max_count=3):
    """동의어/반의어 향상된 필터링 (어근 중복 제거 + 개수 제한)"""

    if not candidates:
        return []

    # 1단계: 기본 필터링
    basic_filtered = []
    for candidate in candidates:
        candidate = str(candidate).strip()

        # 🔥 두 개 이상 단어 강력 차단
        if " " in candidate or "-" in candidate or "_" in candidate:
            continue

        # 🔥 숫자나 특수문자 포함 차단 (아포스트로피 제외)
        if not re.match(r"^[a-zA-Z']+$", candidate):
            continue

        # 🔥 길이 제한
        if len(candidate) < 3 or len(candidate) > 12:
            continue

        # 🔥 원본과 동일한 것 제거
        if candidate.lower() == original_word.lower():
            continue

        basic_filtered.append(candidate)

    # 2단계: 어근 중복 제거
    try:
        from nltk.stem import PorterStemmer

        ps = PorterStemmer()

        seen_stems = set()
        original_stem = ps.stem(original_word.lower())
        seen_stems.add(original_stem)  # 원본 어근 먼저 추가

        unique_filtered = []
        for candidate in basic_filtered:
            stem = ps.stem(candidate.lower())
            if stem not in seen_stems:
                unique_filtered.append(candidate)
                seen_stems.add(stem)

        # 3단계: 개수 제한
        return unique_filtered[:max_count]

    except Exception:
        # NLTK 오류 시 기본 중복 제거 + 개수 제한
        unique_basic = list(dict.fromkeys(basic_filtered))
        return unique_basic[:max_count]


# 전역 변수 초기화
spacy = None
nlp = None
openai = None
client = None
nltk = None
lemmatizer = None

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
    # 감정 동사 (교육적 가치 무관하게 제외)
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


def safe_import_packages():
    """필수 패키지들을 안전하게 import"""
    global spacy, openai, nltk, nlp, lemmatizer, client

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        print(
            "❌ spacy가 설치되지 않았습니다. 'pip install spacy' 실행 후 'python -m spacy download en_core_web_sm' 실행하세요."
        )
        return False
    except OSError:
        print(
            "❌ spacy 영어 모델이 없습니다. 'python -m spacy download en_core_web_sm' 실행하세요."
        )
        return False

    try:
        import openai

        # OpenAI API 키 검증
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            print("💡 다음 중 한 가지 방법으로 설정하세요:")
            print("   1. export OPENAI_API_KEY='your-api-key-here'")
            print("   2. .env 파일에 OPENAI_API_KEY=your-api-key-here 추가")
            print("   3. 시스템 환경변수로 설정")
            return False

        try:
            client = openai.OpenAI(api_key=api_key)
            # API 키 유효성 간단 테스트
            test_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            print("✅ OpenAI API 연결 및 인증 성공")
        except openai.AuthenticationError:
            print("❌ OpenAI API 키가 유효하지 않습니다. API 키를 확인해주세요.")
            return False
        except openai.RateLimitError:
            print("⚠️ OpenAI API 사용량 한도에 도달했습니다. 잠시 후 다시 시도하세요.")
            return False
        except openai.APIError as e:
            print(f"❌ OpenAI API 오류: {e}")
            return False
        except Exception as e:
            print(f"❌ OpenAI 초기화 실패: {e}")
            return False

    except ImportError:
        print("❌ openai가 설치되지 않았습니다. 'pip install openai' 실행하세요.")
        return False

    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()

        # NLTK 데이터 확인
        required_nltk_data = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/wordnet", "wordnet"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/stopwords", "stopwords"),
        ]

        for data_path, download_name in required_nltk_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                print(f"📦 NLTK {download_name} 데이터 다운로드 중...")
                try:
                    nltk.download(download_name, quiet=True)
                except Exception as e:
                    print(f"⚠️ NLTK {download_name} 다운로드 실패: {e}")
                    print("인터넷 연결을 확인하거나 수동으로 다운로드하세요.")

    except ImportError:
        print("❌ nltk가 설치되지 않았습니다. 'pip install nltk' 실행하세요.")
        return False

    return True


def safe_gpt_call(
    client,
    prompt_messages,
    model="gpt-4o",
    max_tokens=300,
    temperature=0.1,
    max_retries=3,
):
    """안전한 GPT API 호출 with 재시도 로직"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response, None

        except openai.AuthenticationError as e:
            error_msg = "API 키 인증 실패"
            print(f"❌ {error_msg}: {e}")
            return None, error_msg

        except openai.RateLimitError as e:
            wait_time = min(2**attempt, 60)  # 지수 백오프, 최대 60초
            print(
                f"⚠️ API 사용량 한도 도달. {wait_time}초 대기 후 재시도... (시도 {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)
            continue

        except openai.APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"⚠️ API 오류 발생. {wait_time}초 대기 후 재시도... (시도 {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"API 오류 (최대 재시도 횟수 초과): {e}"
                print(f"❌ {error_msg}")
                return None, error_msg

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"⚠️ 예상치 못한 오류. {wait_time}초 대기 후 재시도... (시도 {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"예상치 못한 오류 (최대 재시도 횟수 초과): {e}"
                print(f"❌ {error_msg}")
                return None, error_msg

    return None, "최대 재시도 횟수 초과"


# 패키지 import 실행
if not safe_import_packages():
    print("❌ 필수 패키지 import 실패. 프로그램을 종료합니다.")
    exit(1)


# 🔥 고등학생 수준 쉬운 단어 판별 함수
def enhanced_is_too_easy_for_highschool(word, pos, easy_words, child_vocab, freq_tiers):
    """고등학생 수준에서 너무 쉬운 단어인지 판별 - GPT 필터와 함께 사용"""
    word_lower = word.lower()

    # 1. 기본 단어 강력 필터링 추가 (맨 앞에)
    if (
        word_lower in BASIC_VERBS
        or word_lower in BASIC_ADJECTIVES
        or word_lower in BASIC_NOUNS
    ):
        return True

    # 2. 외부 DB 체크 추가
    if is_basic_by_external_db(word_lower):
        return True

    # 3. 아동 어휘 체크
    if child_vocab and word_lower in child_vocab:
        return True

    # 4. 매우 짧은 단어
    if len(word_lower) <= 2:
        return True

    # 5. 기본 문법 요소
    if pos in {"DET", "ADP", "CONJ", "PRON", "AUX", "INTJ"}:
        return True

    return False


# 🔥 contextual_meaning.py 직접 통합
def integrated_get_best_korean_definition(
    word,
    phrase_db=None,
    is_phrase=False,
    max_tokens=10000,
    client=None,
    gpt_cache=None,
    gpt_call_count=0,
    GPT_CALL_LIMIT=100,
    token_usage=None,
    custom_prompt=None,
    sentence="",  # 🔥 추가
):
    """contextual_meaning.py의 get_best_korean_definition 통합 버전"""

    # 전역 변수 초기화
    if gpt_cache is None:
        gpt_cache = {}
    if token_usage is None:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    word_lower = word.lower()

    # 1. phrase_db에서 먼저 찾기
    if phrase_db is not None:
        found = phrase_db[phrase_db["phrase"].str.lower() == word_lower]
        if not found.empty:
            kor = found.iloc[0].get("definition", "")
            if kor and re.search("[가-힣]", kor):
                return kor.strip(), gpt_cache, gpt_call_count, token_usage

    # 2. 토큰 제한 확인
    if token_usage["total_tokens"] >= max_tokens:
        print(f"      ❌ 토큰 제한 도달")
        return word, gpt_cache, gpt_call_count, token_usage

    # 3. GPT 호출
    if gpt_call_count >= GPT_CALL_LIMIT:
        print(f"      ❌ GPT 호출 제한 도달")
        return word, gpt_cache, gpt_call_count, token_usage

    # 캐시 확인
    cache_key = (word, is_phrase)
    if cache_key in gpt_cache:
        return gpt_cache[cache_key], gpt_cache, gpt_call_count, token_usage

    # GPT 호출
    if custom_prompt:
        prompt = custom_prompt
    else:
        # 문맥 정보가 있으면 문맥 특화 프롬프트
        if sentence and len(sentence.strip()) > 10:
            prompt = f"""Analyze the word "{word}" in this specific context and provide the Korean meaning that fits THIS usage.

Context sentence: "{sentence}"

Provide the Korean meaning for how "{word}" is used in THIS specific context:
- Focus on the meaning in this particular sentence
- Not the general dictionary meaning
- Korean translation should match this specific usage

Korean meaning for this context:"""
        else:
            # 기존 사전적 의미 프롬프트 유지
            prompt = f"""Provide the accurate Korean meaning of the following English {'idiom' if is_phrase else 'word'}.
            
    English: "{word}"
    Korean Translation:"""

        system_message = "You are an expert English-Korean translator. Return only Korean meanings without examples."

    # GPT 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        answer = response.choices[0].message.content.strip().replace('"', "")
        gpt_call_count += 1

        # 토큰 사용량 업데이트
        if hasattr(response, "usage"):
            usage = response.usage
            token_usage["prompt_tokens"] += usage.prompt_tokens
            token_usage["completion_tokens"] += usage.completion_tokens
            token_usage["total_tokens"] += usage.total_tokens

        gpt_cache[cache_key] = answer
        return answer, gpt_cache, gpt_call_count, token_usage

    except Exception as e:
        print("GPT 호출 실패:", e)
        return word, gpt_cache, gpt_call_count, token_usage


# 🔥 vocab_difficulty.py 핵심 함수들 직접 통합
def integrated_extract_info(word, pos=None):
    """vocab_difficulty.py의 extract_info 통합 버전"""
    definition = ""
    synonyms = []
    antonyms = []

    # 품사 태그가 없으면 추론
    if not pos:
        try:
            from nltk import pos_tag

            tagged = pos_tag([word])
            pos = integrated_get_wordnet_pos(tagged[0][1]) if tagged else None
        except:
            pos = None

    # WordNet에서 정보 추출
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)

    if synsets:
        # 첫 번째 동의어 집합의 정의 사용
        definition = synsets[0].definition()

        # 동의어 수집
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name().lower() != word.lower():
                    synonyms.append(lemma.name().replace("_", " "))

                # 반의어 수집
                for antonym in lemma.antonyms():
                    antonyms.append(antonym.name().replace("_", " "))

    # 중복 제거
    synonyms = list(set(synonyms))[:5]
    antonyms = list(set(antonyms))[:5]

    return definition, synonyms, antonyms


def integrated_get_wordnet_pos(tag):
    """vocab_difficulty.py의 get_wordnet_pos 통합 버전"""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def integrated_calculate_phonetic_complexity(word):
    """vocab_difficulty.py의 calculate_phonetic_complexity 통합 버전"""
    vowels = set("aeiou")
    consonants = set("bcdfghjklmnpqrstvwxyz")
    complexity = 0
    consonant_clusters = 0
    prev_consonant = False

    for char in word.lower():
        if char in consonants:
            if prev_consonant:
                consonant_clusters += 1
            prev_consonant = True
        else:
            prev_consonant = False

    complexity += consonant_clusters * 0.5

    rare_combinations = ["ph", "th", "ch", "sh", "gh", "ck", "ng", "qu"]
    for combo in rare_combinations:
        if combo in word.lower():
            complexity += 0.3

    return complexity


def integrated_get_word_difficulty_score(word, nlp_model=None):
    """vocab_difficulty.py의 get_word_difficulty_score 통합 버전"""
    try:
        # 기본 점수 계산
        morphological_complexity = 0

        if nlp_model:
            doc = nlp_model(word)
            for token in doc:
                if token.lemma_ != token.text:
                    morphological_complexity += 1
                if token.pos_ in ["NOUN", "VERB"]:
                    morphological_complexity += 0.5
                elif token.pos_ in ["ADJ", "ADV"]:
                    morphological_complexity += 1

        # WordNet 기반 의미 복잡도
        synsets = wordnet.synsets(word)
        semantic_complexity = len(synsets) * 0.1

        # 음성적 복잡도
        phonetic_complexity = integrated_calculate_phonetic_complexity(word)

        # 총 점수 계산
        total_score = (
            morphological_complexity * 0.4
            + semantic_complexity * 0.3
            + phonetic_complexity * 0.3
        )

        return total_score
    except:
        return 0.5  # 기본값


def integrated_is_difficult_word(
    word,
    easy_words,
    children_vocab,
    frequency_tiers=None,
    nlp_model=None,
    threshold=2.8,
):
    """vocab_difficulty.py의 is_difficult_word 통합 버전"""
    min_length = 4

    if not word or len(word) < min_length:
        return False

    if not word.isalpha():
        return False

    # 쉬운 단어 체크
    if word.lower() in easy_words:
        return False

    if children_vocab and word.lower() in children_vocab:
        return False

    # WordNet 기반 기본 판별
    synsets = wordnet.synsets(word)
    if not synsets:
        return True  # 사전에 없는 단어는 어려운 것으로 간주

    # 난이도 점수 기반 판별
    difficulty_score = integrated_get_word_difficulty_score(word, nlp_model)
    return difficulty_score > 0.6


def load_custom_idioms_from_data_directory(data_dir: str = "data") -> list:
    """data 디렉토리에서 숙어 로드"""
    idioms = set()
    loading_summary = []

    print(f"📁 data 디렉토리 숙어 로딩 시작: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"❌ 디렉토리가 존재하지 않습니다: {data_dir}")
        return []

    file_patterns = ["*.csv", "*.txt", "*.xlsx", "*.xls"]
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(os.path.join(data_dir, pattern)))

    print(f"🔍 발견된 파일 수: {len(all_files)}개")

    for file_path in all_files:
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        file_idioms_count = 0

        try:
            print(f"📄 처리 중: {filename}")

            if file_ext == ".csv":
                df = pd.read_csv(file_path, encoding="utf-8")
                possible_columns = [
                    "phrase",
                    "idiom",
                    "expression",
                    "text",
                    "content",
                    "원형",
                ]
                target_column = next(
                    (
                        col
                        for col in df.columns
                        if col.lower() in [c.lower() for c in possible_columns]
                    ),
                    df.columns[0] if len(df.columns) > 0 else None,
                )

                if target_column and target_column in df.columns:
                    phrases = (
                        df[target_column].dropna().astype(str).str.strip().str.lower()
                    )
                    phrases = phrases[phrases != ""].unique()
                    valid_phrases = [
                        phrase
                        for phrase in phrases
                        if 2 <= len(phrase.split()) <= 8
                        and len(phrase) <= 100
                        and phrase.replace(" ", "").replace("-", "").isalpha()
                    ]
                    idioms.update(valid_phrases)
                    file_idioms_count = len(valid_phrases)
                    print(
                        f"   ✅ CSV: {file_idioms_count}개 추출 (컬럼: {target_column})"
                    )

            elif file_ext == ".txt":
                encodings = ["utf-8", "cp949", "euc-kr", "latin1"]
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            lines = f.readlines()
                        phrases = [
                            line.strip().lower()
                            for line in lines
                            if line.strip()
                            and not line.startswith(("#", "//"))
                            and 2 <= len(line.split()) <= 8
                            and len(line) <= 100
                        ]
                        idioms.update(phrases)
                        file_idioms_count = len(phrases)
                        print(
                            f"   ✅ TXT: {file_idioms_count}개 추출 (인코딩: {encoding})"
                        )
                        break
                    except UnicodeDecodeError:
                        continue

            elif file_ext in [".xlsx", ".xls"]:
                excel_file = pd.ExcelFile(file_path)
                total_phrases = []
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        possible_columns = [
                            "phrase",
                            "idiom",
                            "expression",
                            "text",
                            "content",
                            "원형",
                        ]
                        target_column = next(
                            (
                                col
                                for col in df.columns
                                if col.lower() in [c.lower() for c in possible_columns]
                            ),
                            df.columns[0] if len(df.columns) > 0 else None,
                        )
                        if target_column and target_column in df.columns:
                            phrases = (
                                df[target_column]
                                .dropna()
                                .astype(str)
                                .str.strip()
                                .str.lower()
                            )
                            phrases = phrases[phrases != ""].unique()
                            valid_phrases = [
                                phrase
                                for phrase in phrases
                                if 2 <= len(phrase.split()) <= 8
                                and len(phrase) <= 100
                                and phrase.replace(" ", "").replace("-", "").isalpha()
                            ]
                            total_phrases.extend(valid_phrases)
                            print(f"      📋 {sheet_name}: {len(valid_phrases)}개")
                    except Exception as e:
                        print(f"      ❌ {sheet_name} 시트 처리 실패: {e}")
                idioms.update(total_phrases)
                file_idioms_count = len(total_phrases)
                print(f"   ✅ Excel: 총 {file_idioms_count}개 추출")

            loading_summary.append(
                {
                    "file": filename,
                    "type": file_ext,
                    "loaded_count": file_idioms_count,
                    "status": "success" if file_idioms_count > 0 else "empty",
                }
            )

        except Exception as e:
            print(f"   ❌ {filename} 처리 실패: {e}")
            loading_summary.append(
                {
                    "file": filename,
                    "type": file_ext,
                    "loaded_count": 0,
                    "status": "failed",
                    "error": str(e),
                }
            )

    final_idioms = sorted(set(idioms))
    print(f"\n📊 data 디렉토리 로딩 완료: 총 {len(final_idioms)}개 숙어")
    return final_idioms


class SafeCacheManager:
    """Windows 호환 간단한 캐시 관리자"""

    def __init__(self, cache_dir: str = "cache", app_name: str = "vocab_extractor"):
        self.cache_dir = cache_dir
        self.app_name = app_name
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_filename(self, cache_type: str, identifier: str = "") -> str:
        """캐시 파일명 생성"""
        if identifier:
            import hashlib

            hash_id = hashlib.md5(identifier.encode()).hexdigest()[:8]
            filename = f"{self.app_name}_{cache_type}_{hash_id}.json"
        else:
            filename = f"{self.app_name}_{cache_type}.json"
        return os.path.join(self.cache_dir, filename)

    def load_cache(self, cache_type: str, identifier: str = "") -> Dict[str, Any]:
        """캐시 로드"""
        cache_file = self.get_cache_filename(cache_type, identifier)
        try:
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # 간단한 유효성 검사
                if isinstance(cache_data, dict) and "data" in cache_data:
                    print(f"✅ 캐시 로드 성공: {cache_type}")
                    return cache_data["data"]
                elif isinstance(cache_data, dict):
                    return cache_data
            return {}
        except Exception as e:
            print(f"⚠️ 캐시 로드 실패 ({cache_type}): {e}")
            return {}

    def save_cache(self, cache_type: str, data: Dict[str, Any], identifier: str = ""):
        """캐시 저장"""
        cache_file = self.get_cache_filename(cache_type, identifier)
        try:
            cache_data = {
                "metadata": {
                    "created_at": time.time(),
                    "app_name": self.app_name,
                    "cache_type": cache_type,
                },
                "data": data,
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            print(f"✅ 캐시 저장 성공: {cache_type}")

        except Exception as e:
            print(f"⚠️ 캐시 저장 실패 ({cache_type}): {e}")

    def clear_cache(self, cache_type: str = None):
        """캐시 삭제"""
        try:
            if cache_type:
                cache_file = self.get_cache_filename(cache_type)
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    print(f"✅ 캐시 삭제: {cache_type}")
            else:
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith(self.app_name):
                        os.remove(os.path.join(self.cache_dir, filename))
                print(f"✅ 모든 캐시 삭제 완료")
        except Exception as e:
            print(f"⚠️ 캐시 삭제 실패: {e}")


# 통합 캐시 관리자 (두 파일에서 공통 사용)
class UnifiedCacheManager:
    """improved_vocab_extractor.py와 quality_checker.py에서 공통 사용할 캐시 관리자"""

    def __init__(self):
        self.cache_manager = SafeCacheManager(
            cache_dir="cache", app_name="vocab_system"
        )

        # 캐시 타입 정의
        self.CACHE_TYPES = {
            "gpt_difficulty": "gpt_difficulty",
            "gpt_quality": "gpt_quality",
            "gpt_contextual": "gpt_contextual",
            "separable_analysis": "separable_analysis",
            "user_words": "user_words",
            "easy_words": "easy_words",
        }

    def load_gpt_cache(
        self, cache_type: str, user_identifier: str = ""
    ) -> Dict[str, Any]:
        """GPT 캐시 로드 (타입별)"""
        return self.cache_manager.load_cache(cache_type, user_identifier)

    def save_gpt_cache(
        self, cache_type: str, data: Dict[str, Any], user_identifier: str = ""
    ):
        """GPT 캐시 저장 (타입별)"""
        self.cache_manager.save_cache(cache_type, data, user_identifier)

    def merge_caches(
        self, cache_type: str, new_data: Dict[str, Any], user_identifier: str = ""
    ):
        """기존 캐시와 새 데이터 병합"""
        try:
            existing_cache = self.load_gpt_cache(cache_type, user_identifier)

            # 기존 데이터와 병합 (새 데이터가 우선)
            merged_data = {**existing_cache, **new_data}

            self.save_gpt_cache(cache_type, merged_data, user_identifier)

            print(
                f"✅ 캐시 병합 완료: {cache_type} (기존 {len(existing_cache)}개 + 신규 {len(new_data)}개 = 총 {len(merged_data)}개)"
            )

        except Exception as e:
            print(f"⚠️ 캐시 병합 실패 ({cache_type}): {e}")

    def cleanup_old_caches(self, days_old: int = 7):
        """오래된 캐시 정리"""
        try:
            from datetime import datetime, timedelta

            cutoff_date = datetime.now() - timedelta(days=days_old)
            removed_count = 0

            for filename in os.listdir(self.cache_manager.cache_dir):
                if filename.startswith(
                    self.cache_manager.app_name
                ) and filename.endswith(".json"):
                    filepath = os.path.join(self.cache_manager.cache_dir, filename)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

                    if file_mtime < cutoff_date:
                        os.remove(filepath)
                        removed_count += 1

            if removed_count > 0:
                print(f"✅ 오래된 캐시 {removed_count}개 삭제 완료 ({days_old}일 이상)")

        except Exception as e:
            print(f"⚠️ 캐시 정리 실패: {e}")

    def get_unified_stats(self) -> Dict[str, Any]:
        """통합 캐시 통계"""
        return self.cache_manager.get_cache_stats()


# 전역 캐시 관리자 인스턴스
_global_cache_manager = None


def get_cache_manager() -> UnifiedCacheManager:
    """전역 캐시 관리자 인스턴스 반환 (싱글톤 패턴)"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = UnifiedCacheManager()
    return _global_cache_manager


class SeparableIdiomDetector:
    """사용자 DB의 분리형 숙어 감지기 (OpenAI 활용)"""

    def __init__(self, client, verbose=False):
        self.client = client
        self.verbose = verbose
        self.separable_cache = {}  # GPT 호출 결과 캐싱
        self.gpt_calls = 0

        # 분리형 숙어 데이터베이스
        self.user_separable_idioms = {}  # {원본숙어: 분리형정보}
        self.separable_patterns = {}  # {원본숙어: [패턴들]}

        print("🔧 사용자 DB 분리형 숙어 감지기 초기화 완료")

    def analyze_user_idioms_with_gpt(self, user_idioms: Set[str]) -> Dict[str, Dict]:
        """GPT로 사용자 DB 숙어들의 분리형 가능성 분석"""

        print(f"🤖 GPT로 사용자 숙어 분리형 분석 시작: {len(user_idioms)}개")

        results = {}

        # 띄어쓰기가 있는 숙어들만 분석
        potential_idioms = [
            idiom for idiom in user_idioms if " " in idiom and len(idiom.split()) == 2
        ]

        print(f"   📋 분석 대상 2단어 숙어: {len(potential_idioms)}개")

        for idiom in potential_idioms:
            if idiom in self.separable_cache:
                results[idiom] = self.separable_cache[idiom]
                continue

            try:
                analysis = self._gpt_analyze_separable_idiom(idiom)
                results[idiom] = analysis
                self.separable_cache[idiom] = analysis

                if self.verbose and analysis.get("is_separable", False):
                    display = analysis.get("display_form", idiom)
                    print(f"      ✅ 분리형 확인: {idiom} → {display}")

            except Exception as e:
                if self.verbose:
                    print(f"      ❌ GPT 분석 실패 ({idiom}): {e}")
                continue

        # 분리형 숙어만 필터링
        separable_count = len(
            [r for r in results.values() if r.get("is_separable", False)]
        )
        print(f"   🎯 분리형 숙어 발견: {separable_count}개")

        return results

    def _gpt_analyze_separable_idiom(self, idiom: str) -> Dict:
        """GPT로 개별 숙어의 분리형 여부 분석"""

        words = idiom.split()
        if len(words) != 2:
            return {"is_separable": False, "reason": "Not a two-word phrase"}

        verb, particle = words[0], words[1]

        prompt = f"""
Analyze the English phrasal verb "{idiom}":

Please determine if this is a separable phrasal verb.

Characteristics of separable phrasal verbs:
1. Structure: verb + particle (up, down, on, off, out, in, away, back, etc.)
2. When there's an object, it can be placed between the verb and particle
   Example: pick up → pick something up, pick it up
3. Pronoun objects (it, him, her, them) MUST be placed between verb and particle
   Example: pick it up (✓), pick up it (✗)

Analyze "{idiom}":
- Is "{verb}" a verb?
- Is "{particle}" a particle?
- Can an object be placed between them?
- Is this actually used as a separable phrasal verb in real English?

Please respond in JSON format:
{{
    "is_separable": true/false,
    "verb": "{verb}",
    "particle": "{particle}", 
    "display_form": "if separable, show as 'verb ~ particle' format",
    "examples": ["examples of separable usage"],
    "reason": "reasoning for the decision",
    "confidence": 0.0-1.0
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in English phrasal verbs. You can accurately identify separable phrasal verbs and their usage patterns.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )

            self.gpt_calls += 1
            content = response.choices[0].message.content.strip()

            # JSON 파싱
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()

            result = json.loads(content)

            # 결과 검증 및 보완
            if not isinstance(result.get("is_separable"), bool):
                result["is_separable"] = False

            if result["is_separable"] and not result.get("display_form"):
                result["display_form"] = f"{verb} ~ {particle}"

            return result

        except Exception as e:
            if self.verbose:
                print(f"GPT 분석 오류: {e}")
            return {
                "is_separable": False,
                "reason": f"GPT 분석 실패: {str(e)}",
                "confidence": 0.0,
            }

    def build_separable_patterns(self, separable_analysis: Dict[str, Dict]):
        """분리형 숙어들의 감지 패턴 생성"""

        print(f"🔧 분리형 숙어 패턴 생성 중...")

        for idiom, analysis in separable_analysis.items():
            if not analysis.get("is_separable", False):
                continue

            words = idiom.split()
            if len(words) != 2:
                continue

            verb, particle = words[0], words[1]
            display_form = analysis.get("display_form", f"{verb} ~ {particle}")

            # 다양한 분리형 패턴들 생성
            patterns = self._generate_detection_patterns(verb, particle)

            self.user_separable_idioms[idiom] = {
                "display_form": display_form,
                "verb": verb,
                "particle": particle,
                "confidence": analysis.get("confidence", 0.9),
                "examples": analysis.get("examples", []),
                "reason": analysis.get("reason", ""),
            }

            self.separable_patterns[idiom] = patterns

            if self.verbose:
                print(f"   ✅ {idiom} → {display_form} ({len(patterns)}개 패턴)")

        print(f"✅ 분리형 패턴 생성 완료: {len(self.user_separable_idioms)}개 숙어")

    def _generate_detection_patterns(self, verb: str, particle: str) -> List[Dict]:
        """개별 분리형 숙어의 감지 패턴들 생성"""

        patterns = []

        # 1. 연속형 (기본형)
        patterns.append(
            {
                "pattern": rf"\b{re.escape(verb)}\s+{re.escape(particle)}\b",
                "type": "continuous",
                "description": "연속형",
                "is_separated": False,
                "priority": 1,
            }
        )

        # 2. 분리형 - 일반 명사/명사구
        patterns.append(
            {
                "pattern": rf"\b{re.escape(verb)}\s+(?:the\s+|a\s+|an\s+|this\s+|that\s+|his\s+|her\s+|my\s+|your\s+|our\s+|their\s+)?(?:\w+\s+)*\w+\s+{re.escape(particle)}\b",
                "type": "separated_noun",
                "description": "분리형(명사)",
                "is_separated": True,
                "priority": 2,
            }
        )

        # 3. 분리형 - 대명사 (반드시 분리)
        patterns.append(
            {
                "pattern": rf"\b{re.escape(verb)}\s+(?:it|him|her|them|this|that|these|those)\s+{re.escape(particle)}\b",
                "type": "separated_pronoun",
                "description": "분리형(대명사)",
                "is_separated": True,
                "priority": 3,
            }
        )

        # 4. 분리형 - 긴 명사구
        patterns.append(
            {
                "pattern": rf"\b{re.escape(verb)}\s+(?:the\s+)?(?:\w+\s+){{2,}}\w+\s+{re.escape(particle)}\b",
                "type": "separated_long_noun",
                "description": "분리형(긴명사구)",
                "is_separated": True,
                "priority": 2,
            }
        )

        return patterns

    def detect_separable_idioms_in_text(self, text: str) -> List[Dict]:
        """텍스트에서 사용자 DB 분리형 숙어 감지"""

        if not self.user_separable_idioms:
            return []

        results = []
        found_positions = set()

        if self.verbose:
            print(f"🔍 분리형 숙어 감지 중: {len(self.user_separable_idioms)}개 대상")

        # 우선순위순으로 정렬 (분리형을 먼저 찾아서 연속형과 중복 방지)
        for idiom, info in self.user_separable_idioms.items():
            patterns = self.separable_patterns.get(idiom, [])

            # 우선순위순으로 패턴 정렬 (분리형 먼저)
            sorted_patterns = sorted(
                patterns, key=lambda x: x["priority"], reverse=True
            )

            for pattern_info in sorted_patterns:
                pattern = pattern_info["pattern"]

                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    start, end = match.span()

                    # 중복 위치 확인
                    if any(abs(start - pos[0]) <= 5 for pos in found_positions):
                        continue

                    matched_text = match.group().strip()

                    # 결과 생성
                    result = {
                        "original": matched_text,
                        "base_form": idiom,
                        "display_form": info["display_form"],
                        "pattern_type": pattern_info["type"],
                        "description": pattern_info["description"],
                        "is_separated": pattern_info["is_separated"],
                        "confidence": info["confidence"],
                        "start": start,
                        "end": end,
                        "separable_info": {
                            "verb": info["verb"],
                            "particle": info["particle"],
                            "detection_pattern": pattern,
                            "gpt_reason": info.get("reason", ""),
                            "examples": info.get("examples", []),
                        },
                    }

                    results.append(result)
                    found_positions.add((start, end))

                    if self.verbose:
                        sep_mark = "🔧" if result["is_separated"] else "📝"
                        print(
                            f"      {sep_mark} 발견: '{matched_text}' → {info['display_form']} ({pattern_info['description']})"
                        )

                    # 같은 숙어의 다른 패턴은 스킵 (첫 번째 매치만)
                    break

        return results

    def integrate_with_extractor(self, extractor_instance):
        """기존 AdvancedVocabExtractor와 통합"""

        print(f"🔗 기존 extractor와 분리형 숙어 감지기 통합...")

        # 기존 사용자 숙어들 분석
        if hasattr(extractor_instance, "user_idioms"):
            separable_analysis = self.analyze_user_idioms_with_gpt(
                extractor_instance.user_idioms
            )
            self.build_separable_patterns(separable_analysis)

            # extractor에 분리형 정보 추가
            extractor_instance.separable_detector = self
            extractor_instance.user_separable_idioms = self.user_separable_idioms

            print(
                f"✅ 통합 완료: {len(self.user_separable_idioms)}개 분리형 숙어 활성화"
            )
        else:
            print("⚠️ extractor에 user_idioms가 없습니다")

    def save_separable_analysis(self, output_file: str = "separable_analysis.json"):
        """분리형 분석 결과 저장"""

        analysis_data = {
            "separable_idioms": self.user_separable_idioms,
            "total_count": len(self.user_separable_idioms),
            "gpt_calls": self.gpt_calls,
            "cache": self.separable_cache,
        }

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 분리형 분석 결과 저장: {output_file}")
        except Exception as e:
            print(f"❌ 저장 실패: {e}")

    def load_separable_analysis(self, input_file: str = "separable_analysis.json"):
        """기존 분리형 분석 결과 로드"""

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)

            self.user_separable_idioms = analysis_data.get("separable_idioms", {})
            self.separable_cache = analysis_data.get("cache", {})

            # 패턴 재생성
            for idiom, info in self.user_separable_idioms.items():
                verb = info.get("verb", "")
                particle = info.get("particle", "")
                if verb and particle:
                    patterns = self._generate_detection_patterns(verb, particle)
                    self.separable_patterns[idiom] = patterns

            print(f"✅ 분리형 분석 결과 로드: {len(self.user_separable_idioms)}개")
            return True

        except FileNotFoundError:
            print(f"📂 {input_file} 파일이 없습니다. 새로 분석합니다.")
            return False
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return False


class ExternalVocabDatabase:
    def __init__(self):
        self.oxford_3000 = self._load_oxford_3000()
        self.coca_2000 = self._load_coca_frequent()
        self.gsl_2000 = self._load_general_service_list()

    def _load_oxford_3000(self):
        """Oxford 3000 기본 어휘 로드"""
        try:
            # 로컬 파일이 있으면 로드, 없으면 기본 세트 반환
            oxford_file = "data/oxford_3000.txt"
            if os.path.exists(oxford_file):
                with open(oxford_file, "r", encoding="utf-8") as f:
                    return set(line.strip().lower() for line in f if line.strip())
            else:
                # 기본 Oxford 기초 단어들
                return {
                    "about",
                    "above",
                    "across",
                    "act",
                    "active",
                    "activity",
                    "add",
                    "afraid",
                    "after",
                    "again",
                    "against",
                    "age",
                    "ago",
                    "agree",
                    "air",
                    "all",
                    "alone",
                    "along",
                    "already",
                    "although",
                    "always",
                    "among",
                    "angry",
                    "animal",
                    "answer",
                    "any",
                    "anyone",
                    "anything",
                    "appear",
                    "area",
                    "argue",
                    "arm",
                    "army",
                    "around",
                    "arrive",
                    "art",
                    "article",
                    "ask",
                    "attack",
                    "aunt",
                    "autumn",
                    "away",
                }
        except:
            return set()

    def _load_coca_frequent(self):
        """COCA 최고빈도 2000개 단어 로드"""
        try:
            coca_file = "data/coca_2000.txt"
            if os.path.exists(coca_file):
                with open(coca_file, "r", encoding="utf-8") as f:
                    return set(line.strip().lower() for line in f if line.strip())
            else:
                # COCA 기본 고빈도 단어들
                return {
                    "the",
                    "be",
                    "to",
                    "of",
                    "and",
                    "a",
                    "in",
                    "that",
                    "have",
                    "i",
                    "it",
                    "for",
                    "not",
                    "on",
                    "with",
                    "he",
                    "as",
                    "you",
                    "do",
                    "at",
                    "this",
                    "but",
                    "his",
                    "by",
                    "from",
                    "they",
                    "she",
                    "or",
                    "an",
                    "will",
                    "my",
                    "one",
                    "all",
                    "would",
                    "there",
                    "their",
                    "what",
                    "so",
                    "up",
                    "out",
                    "if",
                    "about",
                    "who",
                    "get",
                    "which",
                    "go",
                    "me",
                    "when",
                    "make",
                    "can",
                    "like",
                    "time",
                    "no",
                    "just",
                    "him",
                    "know",
                    "take",
                    "people",
                    "into",
                    "year",
                    "your",
                    "good",
                    "some",
                    "could",
                    "them",
                    "see",
                    "other",
                    "than",
                    "then",
                    "now",
                    "look",
                    "only",
                    "come",
                    "its",
                    "over",
                    "think",
                    "also",
                    "back",
                    "after",
                    "use",
                    "two",
                    "how",
                    "our",
                    "work",
                    "first",
                    "well",
                    "way",
                    "even",
                    "new",
                    "want",
                    "because",
                }
        except:
            return set()

    def _load_general_service_list(self):
        """General Service List 기본 2000단어 로드"""
        try:
            gsl_file = "data/gsl_2000.txt"
            if os.path.exists(gsl_file):
                with open(gsl_file, "r", encoding="utf-8") as f:
                    return set(line.strip().lower() for line in f if line.strip())
            else:
                # GSL 기본 단어들 (일부)
                return (
                    BASIC_VERBS.union(BASIC_ADJECTIVES)
                    .union(BASIC_NOUNS)
                    .union(
                        {
                            "the",
                            "be",
                            "to",
                            "of",
                            "and",
                            "a",
                            "in",
                            "that",
                            "have",
                            "i",
                            "it",
                            "for",
                            "not",
                            "on",
                            "with",
                            "he",
                            "as",
                            "you",
                            "do",
                            "at",
                            "this",
                            "but",
                            "his",
                            "by",
                        }
                    )
                )
        except:
            return set()

    def is_basic_word(self, word):
        """외부 DB 기반 기본 단어 판별"""
        word_lower = word.lower()
        return (
            word_lower in self.oxford_3000
            or word_lower in self.coca_2000
            or word_lower in self.gsl_2000
        )


def is_basic_by_external_db(word):
    """외부 어휘 DB 활용한 기본 단어 판별"""
    if not hasattr(is_basic_by_external_db, "db"):
        is_basic_by_external_db.db = ExternalVocabDatabase()
    return is_basic_by_external_db.db.is_basic_word(word)


class AdvancedIdiomChecker:
    """고급 숙어 검증기 - 분리형과 문법 패턴 구분"""

    def __init__(self, nlp_model):
        self.nlp = nlp_model

        # 🔥 연속형 가능 구동사 (붙여서 써도 OK)
        self.optional_separable = {
            "pick up",
            "turn on",
            "turn off",
            "look up",
            "put on",
            "take off",
            "bring up",
            "call up",
            "give up",
            "set up",
            "clean up",
            "fill up",
            "write down",
            "sit down",
            "stand up",
            "wake up",
            "get up",
        }

        # 🔥 반드시 분리되어야 하는 구동사 (목적어 필수)
        self.mandatory_separable = {
            "pick": ["up", "out", "off"],
            "turn": ["down", "up"],
            "put": ["away", "back"],
            "take": ["apart", "down"],
            "figure": ["out"],
            "work": ["out"],
            "point": ["out"],
            "carry": ["out"],
            "bring": ["about"],
        }

        # 🔥 문법 패턴 숙어들 (특정 품사 필수)
        self.grammar_patterns = {
            # V-ing 패턴
            r"\bspend\s+(?:time|money|hours?|days?|years?)\s+(\w+ing)\b": "spend time V-ing",
            r"\bis\s+worth\s+(\w+ing)\b": "be worth V-ing",
            r"\bkeep\s+(?:on\s+)?(\w+ing)\b": "keep V-ing",
            r"\bavoid\s+(\w+ing)\b": "avoid V-ing",
            r"\benjoy\s+(\w+ing)\b": "enjoy V-ing",
            r"\bfinish\s+(\w+ing)\b": "finish V-ing",
            # N + V-ing 패턴
            r"\bprevent\s+(\w+)\s+from\s+(\w+ing)\b": "prevent N from V-ing",
            r"\bstop\s+(\w+)\s+from\s+(\w+ing)\b": "stop N from V-ing",
            # 기타 패턴
            r"\bit\s+takes\s+(\w+)\s+to\s+(\w+)": "it takes N to V",
            r"\bthere\s+is\s+no\s+point\s+in\s+(\w+ing)\b": "there is no point in V-ing",
        }

        # 🔥 알려진 일반 숙어 패턴들
        self.known_phrasal_patterns = {
            r"\bas\s+\w+\s+as\b",
            r"\bin\s+order\s+to\b",
            r"\bas\s+a\s+result\b",
            r"\bon\s+the\s+other\s+hand\b",
            r"\bfor\s+instance\b",
            r"\bin\s+spite\s+of\b",
            r"\bbecause\s+of\b",
            r"\binstead\s+of\b",
            r"\baccording\s+to\b",
        }

    def analyze_phrasal_verb_pattern(self, text, verb_token, particle_token):
        """구동사 패턴 분석 - 연속형 vs 분리형 vs 문법패턴"""
        verb = verb_token.lemma_.lower()
        particle = particle_token.text.lower()
        base_phrasal = f"{verb} {particle}"

        # 🔥 1. 연속형 가능한 구동사인지 확인
        if base_phrasal in self.optional_separable:
            return {
                "pattern_type": "optional_separable",
                "base_form": base_phrasal,
                "display_form": base_phrasal,  # 그냥 연속형으로 표시
                "is_separated": False,
            }

        # 🔥 2. 반드시 분리되어야 하는 구동사인지 확인
        if (
            verb in self.mandatory_separable
            and particle in self.mandatory_separable[verb]
        ):
            return {
                "pattern_type": "mandatory_separable",
                "base_form": base_phrasal,
                "display_form": f"{verb} ~ {particle}",  # ~ 로 표시
                "is_separated": True,
            }

        # 🔥 3. 일반 구동사 (실제 분리되었는지 확인)
        verb_idx = verb_token.i
        particle_idx = particle_token.i

        # 동사와 입자 사이에 다른 토큰이 있는지 확인
        if abs(verb_idx - particle_idx) > 1:
            # 실제로 분리되어 있음
            return {
                "pattern_type": "actually_separated",
                "base_form": base_phrasal,
                "display_form": f"{verb} ~ {particle}",
                "is_separated": True,
            }
        else:
            # 연속으로 붙어있음
            return {
                "pattern_type": "continuous",
                "base_form": base_phrasal,
                "display_form": base_phrasal,
                "is_separated": False,
            }

    def analyze_grammar_pattern(self, text):
        """문법 패턴 분석"""
        results = []

        for pattern_regex, pattern_name in self.grammar_patterns.items():
            matches = re.finditer(pattern_regex, text, re.IGNORECASE)
            for match in matches:
                start, end = match.span()
                original_text = match.group()

                # 매칭된 그룹들 분석
                groups = match.groups()

                # V-ing 패턴 검증
                if "V-ing" in pattern_name and groups:
                    # 마지막 그룹이 실제로 동명사인지 확인
                    last_word = groups[-1]
                    if last_word.endswith("ing"):
                        results.append(
                            {
                                "original": original_text,
                                "pattern_type": "grammar_pattern",
                                "display_form": pattern_name,
                                "base_form": pattern_name,
                                "start": start,
                                "end": end,
                                "is_separated": False,
                            }
                        )

                # 기타 패턴
                else:
                    results.append(
                        {
                            "original": original_text,
                            "pattern_type": "grammar_pattern",
                            "display_form": pattern_name,
                            "base_form": pattern_name,
                            "start": start,
                            "end": end,
                            "is_separated": False,
                        }
                    )

        return results


class VocabularyQualityChecker:
    def __init__(self, vocabulary_file):
        self.vocabulary_file = vocabulary_file
        self.df = (
            pd.read_excel(vocabulary_file)
            if vocabulary_file.endswith(".xlsx")
            else pd.read_csv(vocabulary_file)
        )

    def generate_quality_report(self):
        issues = []
        if "뜻(한글)" in self.df.columns:
            empty_meanings = self.df["뜻(한글)"].isna().sum()
            issues.append(f"의미 누락: {empty_meanings}개")

        if "단어" in self.df.columns:
            duplicate_words = self.df["단어"].duplicated().sum()
            issues.append(f"중복 단어: {duplicate_words}개")

        total_issues = len(issues)
        quality_score = max(0, 100 - total_issues * 10)

        return {
            "quality_score": quality_score,
            "total_issues": total_issues,
            "issues": issues,
            "discovered_patterns": 0,
        }

    def update_vocabulary_with_fixes(self, apply_high_confidence_fixes=True):
        fixed_df = self.df.copy()
        if "뜻(한글)" in fixed_df.columns:
            fixed_df["뜻(한글)"] = (
                fixed_df["뜻(한글)"].astype(str).fillna("의미 확인 필요")
            )

        if "단어" in fixed_df.columns:
            fixed_df = fixed_df.drop_duplicates(subset=["단어"])

        return fixed_df


class GPTDifficultyFilter:
    """GPT 기반 사용자 DB 수준 난이도 필터"""

    def __init__(
        self, client, user_words=None, cache_file="difficulty_filter_cache.json"
    ):
        self.client = client
        self.user_words = user_words or set()
        self.cache_file = cache_file
        self.difficulty_cache = {}
        self.gpt_calls = 0
        self._load_cache()

        # 사용자 DB 단어들의 평균 난이도 분석 (한 번만 실행)
        self.user_db_baseline = None
        if self.user_words:
            self._analyze_user_db_baseline()

    def _load_cache(self):
        """캐시 로드"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.difficulty_cache = json.load(f)
                print(f"✅ 난이도 필터 캐시 로드: {len(self.difficulty_cache)}개")
        except Exception as e:
            print(f"⚠️ 캐시 로드 실패: {e}")
            self.difficulty_cache = {}

    def _save_cache(self):
        """캐시 저장"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.difficulty_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")

    def _analyze_user_db_baseline(self):
        """사용자 DB 단어들의 평균 난이도 분석하여 기준점 설정"""
        if not self.user_words or len(self.user_words) == 0:
            return

        print("🔍 사용자 DB 단어 난이도 기준점 분석 중...")

        # 🔥 사용자 DB에서 단일 단어만 선택하여 샘플링 (숙어 제외)
        single_words = [
            word for word in self.user_words if " " not in word and "-" not in word
        ]
        sample_words = single_words[:15] if len(single_words) > 15 else single_words

        if len(sample_words) == 0:
            self.user_db_baseline = {
                "average_score": 5.0,
                "min_threshold": 3.5,
                "sample_count": 0,
            }
            print("   ⚠️ 분석할 단일 단어가 없음, 기본값 사용")
            return

        difficulty_scores = []
        for word in sample_words:
            difficulty = self._gpt_analyze_word_difficulty(word, for_baseline=True)
            if difficulty and "difficulty_score" in difficulty:
                difficulty_scores.append(difficulty["difficulty_score"])

        if difficulty_scores:
            avg_score = sum(difficulty_scores) / len(difficulty_scores)
            # 평균보다 1.5점 낮은 수준까지만 허용, 최소 3.0점은 보장
            min_threshold = max(3.0, avg_score - 1.5)

            self.user_db_baseline = {
                "average_score": avg_score,
                "min_threshold": min_threshold,
                "sample_count": len(difficulty_scores),
            }
            print(
                f"   ✅ 사용자 DB 기준점: 평균 {avg_score:.1f}점, 최소 임계값 {min_threshold:.1f}점"
            )
        else:
            self.user_db_baseline = {
                "average_score": 6.0,
                "min_threshold": 5.5,
                "sample_count": 0,
            }
            print("   ⚠️ 사용자 DB 분석 실패, 기본값 사용")

    def is_word_appropriate_for_user_db(self, word, context="", pos=""):
        """단어가 사용자 DB 수준에 적합한지 GPT로 판별"""

        word_lower = word.lower().strip()

        # 🔥 사용자 DB에 있는 단어는 무조건 적합 (최우선)
        if word_lower in self.user_words:
            return True, "사용자DB포함"

        # 숙어는 별도 처리
        if " " in word or "-" in word or "~" in word:
            return True, "숙어패턴"

        # 캐시 확인
        cache_key = f"{word_lower}:{context[:30]}"
        if cache_key in self.difficulty_cache:
            cached_result = self.difficulty_cache[cache_key]
            return cached_result["appropriate"], cached_result["reason"]

        # GPT 분석 (사용자 DB가 아닌 경우만)
        difficulty_analysis = self._gpt_analyze_word_difficulty(word, context, pos)

        # 적합성 판별
        appropriate, reason = self._determine_appropriateness(word, difficulty_analysis)

        # 캐시 저장
        self.difficulty_cache[cache_key] = {
            "appropriate": appropriate,
            "reason": reason,
            "analysis": difficulty_analysis,
        }

        return appropriate, reason

    def _gpt_analyze_word_difficulty(
        self, word, context="", pos="", for_baseline=False
    ):
        """GPT로 단어 난이도 분석"""

        # 사용자 DB 기준점 정보
        baseline_info = ""
        if self.user_db_baseline and not for_baseline:
            baseline_info = f"""
참고: 현재 사용자 단어 DB의 평균 난이도는 {self.user_db_baseline['average_score']:.1f}점입니다.
이 수준에 맞는 단어들을 선별해주세요.
"""

        prompt = f"""

Analyze the difficulty of the English word "{word}" for Korean high school students.

Context: "{context}"
Part of Speech: {pos}

CRITICAL EXCLUSION CRITERIA (Rate as 1-2 points):
- Basic daily verbs: feel, see, hear, go, come, make, take, get, have, etc.
- Basic adjectives: good, bad, big, small, happy, sad, easy, hard, etc.  
- Basic nouns: man, woman, house, school, work, time, etc.
- Words learned in elementary/middle school
- Common conversation vocabulary

IMPORTANT NOTES:
- Educational value is NOT a factor for inclusion
- Emotional expression importance is NOT a factor  
- Focus ONLY on vocabulary difficulty level
- Rate based on the SPECIFIC meaning in the given context
- Korean high school students already know basic English vocabulary

Difficulty Scale (1-10):
1-3: Elementary/Middle school level (EXCLUDE these)
4-5: Basic high school level (borderline - usually EXCLUDE)  
6-7: Intermediate high school level (INCLUDE only if truly challenging)
8-10: Advanced/University prep level (INCLUDE)

For the word "{word}" in context "{context}":
- What is the specific meaning being used?
- Would a Korean high school student struggle with THIS specific usage?
- Is this an advanced/academic usage of the word?

Respond in JSON format:
{{
    "difficulty_score": 1-10,
    "specific_meaning": "meaning in this context",
    "level_category": "elementary|middle_school|basic_high_school|advanced_high_school|university",
    "is_basic_vocabulary": true/false,
    "recommendation": "include|exclude",
    "reasoning": "focus on difficulty of the specific contextual meaning"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in English education for Korean students. You understand the appropriate vocabulary levels for different stages of English learning.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )

            self.gpt_calls += 1
            content = response.choices[0].message.content.strip()

            # JSON 파싱
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()

            result = json.loads(content)
            return result

        except Exception as e:
            print(f"❌ GPT 난이도 분석 실패 ({word}): {e}")
            return None

    def _determine_appropriateness(self, word, analysis):
        """수정된 적합성 판별 - 더 엄격한 기준"""

        if not analysis:
            return False, "분석실패"

        # 🔥 기본 단어 강력 차단
        if analysis.get("is_basic_vocabulary", False):
            return False, "기본어휘제외"

        difficulty_score = analysis.get("difficulty_score", 5)
        contextual_difficulty = analysis.get("contextual_difficulty", 5)

        # 🔥 대폭 상향된 임계값 (기존 4.0 → 7.0)
        MIN_DIFFICULTY = 6.0
        MIN_CONTEXTUAL_DIFFICULTY = 6.0

        if difficulty_score < 6.0:
            return False, f"난이도부족({difficulty_score}<{MIN_DIFFICULTY})"

        if contextual_difficulty < MIN_CONTEXTUAL_DIFFICULTY:
            return (
                False,
                f"문맥난이도부족({contextual_difficulty}<{MIN_CONTEXTUAL_DIFFICULTY})",
            )

        # 🔥 교육적 가치 기준 완전 제거
        recommendation = analysis.get("recommendation", "exclude")
        if recommendation == "include":
            return (
                True,
                f"고난이도확인({difficulty_score}점,문맥{contextual_difficulty}점)",
            )
        else:
            return False, f"GPT제외추천({analysis.get('reasoning', '이유없음')})"

    def batch_filter_words(self, words_with_context, batch_size=10):
        """여러 단어를 배치로 처리"""

        results = []
        appropriate_count = 0

        print(f"🔍 GPT 기반 단어 적합성 검사: {len(words_with_context)}개")

        for i in range(0, len(words_with_context), batch_size):
            batch = words_with_context[i : i + batch_size]

            for word_info in batch:
                word = word_info.get("word", "")
                context = word_info.get("context", "")
                pos = word_info.get("pos", "")

                appropriate, reason = self.is_word_appropriate_for_user_db(
                    word, context, pos
                )

                results.append(
                    {
                        "word": word,
                        "appropriate": appropriate,
                        "reason": reason,
                        "original_info": word_info,
                    }
                )

                if appropriate:
                    appropriate_count += 1

            # 배치마다 캐시 저장
            if i % (batch_size * 5) == 0:
                self._save_cache()

        # 최종 캐시 저장
        self._save_cache()

        print(f"✅ 필터링 완료: {appropriate_count}/{len(words_with_context)}개 선택")
        print(f"🤖 GPT 호출: {self.gpt_calls}회")

        return results


class AdvancedVocabExtractor(MissingMethodsMixin):

    def __init__(
        self,
        user_words_file="단어DB.csv",
        settings=None,
        csv_file=None,
        verbose=False,
        **kwargs,
    ):
        # 기본 설정
        self.default_settings = {
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

        self.settings = self.default_settings.copy()
        if settings:
            self.settings.update(settings)

        # 기본 변수들
        self.gpt_call_count = 0
        self.GPT_CALL_LIMIT = self.settings["GPT_CALL_LIMIT"]
        self.DIFFICULTY_THRESHOLD = self.settings["DIFFICULTY_THRESHOLD"]
        self.MIN_WORD_LENGTH = self.settings["MIN_WORD_LENGTH"]
        self.MAX_TOKENS = self.settings["MAX_TOKENS"]
        self.USE_CACHE = self.settings["USE_CACHE"]
        self.gpt_cache = {}
        self.gpt_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.verbose = verbose

        # 🔥 사용자 단어 DB 개선된 로딩
        self.user_words = set()
        self.user_idioms = set()  # 🔥 숙어만 따로 관리
        self.user_single_words = set()  # 🔥 단일 단어만 따로 관리

        # Phase 1: 통합 모듈 초기화
        self.phrase_db = None

        # 캐시 로드
        self.load_cache_from_file(self.settings["CACHE_FILE"])

        # 쉬운 단어 목록
        self.easy_words = self._load_easy_words()

        # 기본 컴포넌트들
        self.freq_tiers = {}
        if csv_file and os.path.exists(csv_file):
            print(f"📊 '{csv_file}'에서 빈도 데이터 구축 중...")
            self.freq_tiers = self._build_frequency_from_csv(csv_file)

        # 🔥 client 추가
        self.client = client

        # 🔥 동의어/반의어 추출기 초기화 (새로 추가)
        try:
            # 1순위: 문맥 기반 추출기 시도 (contextual_synonym_refiner.py)
            try:
                from contextual_synonym_refiner import ImprovedSynonymRefiner

                self.synonym_extractor = ImprovedSynonymRefiner(
                    client=client,
                    cache_file="contextual_synonym_cache.json",
                    verbose=self.verbose,
                )
                print("✅ 문맥 기반 동의어/반의어 추출기 초기화 완료")
            except ImportError:
                print("⚠️ contextual_synonym_refiner.py 없음, 기존 모듈 사용")
                # 2순위: 기존 모듈 사용 (synonym_antonym_module.py)
                from synonym_antonym_module import SynonymAntonymExtractor

                self.synonym_extractor = SynonymAntonymExtractor(
                    client=client,
                    cache_file="synonym_antonym_cache.json",
                    verbose=self.verbose,
                )
                print("✅ 기존 동의어/반의어 추출기로 초기화")
        except Exception as e:
            print(f"❌ 동의어/반의어 추출기 초기화 실패: {e}")
            self.synonym_extractor = None
        # 🔥 고급 검증기로 초기화
        self.idiom_checker = AdvancedIdiomChecker(nlp)

        # 🔥 외부 DB 초기화 추가
        self.external_vocab_db = ExternalVocabDatabase()

        # 🔥 사용자 단어 파일 로딩 (개선된 버전)
        if user_words_file and os.path.exists(user_words_file):
            print(f"📖 사용자 단어 파일 로딩: {user_words_file}")
            self._load_user_words_with_idiom_detection(user_words_file)
        else:
            print(f"🔍 사용자 단어 파일을 찾을 수 없습니다: {user_words_file}")

        # 🔥 GPT 기반 난이도 필터 초기화
        self.initialize_gpt_difficulty_filter()
        print(f"✅ 고급 패턴 분석 추출기 초기화 완료")
        print(
            f"   • 통합 컨텍스트 의미 생성: {'✅' if self.settings['USE_INTEGRATED_CONTEXTUAL'] else '❌'}"
        )
        print(
            f"   • 통합 난이도 분석: {'✅' if self.settings['USE_INTEGRATED_DIFFICULTY'] else '❌'}"
        )
        print(f"   • 고급 구동사 분석: ✅")
        print(f"   • 문법 패턴 분석: ✅")

        # 🔥 데이터 소스 로딩 상황
        print(f"\n📊 데이터 소스 로딩 상황:")
        print(f"=" * 60)

        # 🔥 쉬운 단어 로딩 상태
        print(f"\n📚 쉬운 단어 로딩 상황:")
        if os.path.exists(self.settings["EASY_WORDS_CACHE"]):
            print(f"   ✅ 캐시에서 기본 쉬운 단어 로드 완료")
        else:
            print(f"   ⚠️ 캐시 없음, stopwords + 기본 단어들 사용 중")
        print(f"   📊 총 쉬운 단어: {len(self.easy_words)}개")

        # 🔥 data 디렉토리 숙어 로딩
        data_dir = self.settings.get("data_dir", "data")
        self.reference_idioms = load_custom_idioms_from_data_directory(data_dir)

        # 🔥 최종 데이터 요약
        print(f"\n📈 최종 데이터 요약:")
        print(f"   👤 사용자 전체 단어: {len(self.user_words)}개")
        print(f"   📝 사용자 숙어: {len(self.user_idioms)}개")
        print(f"   🔤 사용자 단일 단어: {len(self.user_single_words)}개")
        print(f"   🏛️ 참조 숙어 DB: {len(self.reference_idioms)}개")
        print(f"   📚 쉬운 단어: {len(self.easy_words)}개")
        print(f"   📊 빈도 데이터: {len(self.freq_tiers)}개")
        print(f"=" * 60)

    def initialize_gpt_difficulty_filter(self):
        """GPT 기반 난이도 필터 초기화"""
        if not self.settings.get("USE_GPT_DIFFICULTY_FILTER", True):
            print("⚠️ GPT 난이도 필터가 설정에서 비활성화됨")
            return

        try:
            print("🤖 GPT 기반 난이도 필터 초기화 중...")
            print(f"   📊 사용자 단일 단어: {len(self.user_single_words)}개")

            self.gpt_filter = GPTDifficultyFilter(
                client=client,
                user_words=self.user_single_words,  # 사용자 단일 단어만 사용
                cache_file="gpt_difficulty_filter_cache.json",
            )

            print("✅ GPT 난이도 필터 초기화 완료")

        except Exception as e:
            print(f"⚠️ GPT 난이도 필터 초기화 실패: {e}")
            print("기존 방식으로 계속 진행합니다.")
            self.gpt_filter = None

    # AdvancedVocabExtractor 클래스에 이 메서드 추가
    def debug_gpt_filter_status(self):
        """GPT 필터 상태 디버깅"""
        print("\n🔍 GPT 필터 상태 확인:")
        print(f"   • hasattr(self, 'gpt_filter'): {hasattr(self, 'gpt_filter')}")
        if hasattr(self, "gpt_filter"):
            print(f"   • gpt_filter is not None: {self.gpt_filter is not None}")
            if self.gpt_filter:
                print(f"   • user_words 수: {len(self.gpt_filter.user_words)}")
                print(f"   • baseline 설정: {self.gpt_filter.user_db_baseline}")
                print(f"   • GPT 호출 수: {self.gpt_filter.gpt_calls}")
        print()

    def test_gpt_filter_integration(self):
        """GPT 필터 통합 테스트"""
        print("\n🧪 GPT 필터 통합 테스트:")
        print(
            f"   • gpt_filter 존재: {hasattr(self, 'gpt_filter') and self.gpt_filter is not None}"
        )
        if hasattr(self, "gpt_filter") and self.gpt_filter:
            print(f"   • 사용자 단어 수: {len(self.gpt_filter.user_words)}개")
            print(f"   • 기준점 설정: {self.gpt_filter.user_db_baseline is not None}")
            if self.gpt_filter.user_db_baseline:
                baseline = self.gpt_filter.user_db_baseline
                print(f"   • 평균 점수: {baseline['average_score']:.1f}")
                print(f"   • 최소 임계값: {baseline['min_threshold']:.1f}")
        print("✅ 테스트 완료\n")

    # 3. extract_difficult_words 메서드에서 GPT 필터 사용 확인
    def extract_difficult_words(self, text, easy_words, child_vocab, freq_tiers):
        """사용자 DB 매칭 우선 + GPT 필터링 어려운 단어 추출"""
        text_str = self._force_extract_text(text)
        word_candidates = []

        # 🔥 원형 기준 중복 추적
        seen_lemmas = set()  # 이미 처리된 원형들
        lemma_to_info = {}  # 원형 → 단어 정보 매핑

        try:
            doc = nlp(text_str)
            for token in doc:
                word = token.text.lower()
                lemma = token.lemma_.lower()
                original_word = token.text

                # 기본 필터링
                if (
                    len(word) < 3
                    or not word.isalpha()
                    or token.is_stop
                    or token.pos_ in ["PUNCT", "SPACE", "SYM"]
                ):
                    continue

                # 🔥 원형 기준 중복 체크
                if lemma in seen_lemmas:
                    continue  # 이미 처리된 원형은 건너뛰기

                seen_lemmas.add(lemma)

                # 1차 빠른 필터링
                if word in easy_words or (child_vocab and word in child_vocab):
                    continue

                # 문맥 추출
                context = self._get_sentence_context(
                    text_str, token.idx, token.idx + len(token.text)
                )

                # 후보 단어 수집
                word_candidates.append(
                    {
                        "word": original_word,
                        "lemma": lemma,
                        "context": context,
                        "pos": self._get_simple_pos(token.pos_),
                        "token_info": {
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "original": original_word,
                        },
                    }
                )

        except Exception as e:
            print(f"❌ 토큰 추출 실패: {e}")
            return []

        # 🔥 GPT 필터링 적용 여부 결정 (수정된 조건)
        print(
            f"   🔍 GPT 필터 사용 가능: {hasattr(self, 'gpt_filter') and self.gpt_filter is not None}"
        )

        if hasattr(self, "gpt_filter") and self.gpt_filter and word_candidates:
            print(f"   🤖 GPT 필터링 모드 사용")
            return self._process_words_with_gpt_filter(word_candidates)
        else:
            # 기존 방식으로 처리 (GPT 필터 없음)
            print(f"   ⚙️ 기존 방식 모드 사용")
            return self._process_words_without_gpt_filter(word_candidates, freq_tiers)

    def enhanced_extract_user_db_idioms_with_separable(self, text):
        """분리형 감지 기능이 추가된 사용자 DB 숙어 추출"""

        results = []
        text_str = self._force_extract_text(text)
        found_positions = set()

        print(
            f"   🔍 사용자 DB 숙어 매칭 검사 (분리형 포함): {len(self.user_idioms)}개"
        )

        # 1. 분리형 숙어 감지 (최우선)
        if hasattr(self, "separable_detector"):
            separable_results = self.separable_detector.detect_separable_idioms_in_text(
                text_str
            )

            for sep_result in separable_results:
                context = self._get_sentence_context(
                    text_str, sep_result["start"], sep_result["end"]
                )
                meaning = self.enhanced_korean_definition(
                    sep_result["display_form"], context, is_phrase=True
                )

                results.append(
                    {
                        "original": sep_result["original"],
                        "base_form": sep_result["display_form"],  # 🔥 분리형 표시 포함
                        "meaning": meaning,
                        "context": context,
                        "type": "user_db_separable_idiom",
                        "is_separated": sep_result["is_separated"],
                        "confidence": sep_result["confidence"],
                        "user_db_match": True,
                        "match_type": f"사용자DB분리형_{sep_result['description']}",
                        "separable_info": sep_result["separable_info"],
                    }
                )

                found_positions.add((sep_result["start"], sep_result["end"]))

        # 2. 일반 숙어 추출 (분리형이 아닌 것들)
        non_separable_idioms = self.user_idioms
        if hasattr(self, "user_separable_idioms"):
            non_separable_idioms = self.user_idioms - set(
                self.user_separable_idioms.keys()
            )

        # 길이순 정렬 (긴 숙어부터 매칭하여 중복 방지)
        sorted_regular_idioms = sorted(non_separable_idioms, key=len, reverse=True)

        for idiom in sorted_regular_idioms:
            # 정확한 매칭 (단어 경계 고려)
            pattern = r"\b" + re.escape(idiom) + r"\b"
            matches = re.finditer(pattern, text_str, re.IGNORECASE)

            for match in matches:
                start, end = match.span()

                # 위치 중복 확인
                if any(abs(start - pos[0]) < 5 for pos in found_positions):
                    continue

                context = self._get_sentence_context(text_str, start, end)
                original_text = text_str[start:end]

                meaning = self.enhanced_korean_definition(
                    idiom, context, is_phrase=True
                )

                results.append(
                    {
                        "original": original_text,
                        "base_form": idiom,
                        "meaning": meaning,
                        "context": context,
                        "type": "user_db_idiom",
                        "is_separated": False,
                        "confidence": 0.95,
                        "user_db_match": True,
                        "match_type": "사용자DB일반숙어",
                    }
                )
                found_positions.add((start, end))

        # 결과 통계
        separable_count = len(
            [r for r in results if r.get("type") == "user_db_separable_idiom"]
        )
        regular_count = len([r for r in results if r.get("type") == "user_db_idiom"])

        print(
            f"   📊 사용자 DB 매칭 결과: 분리형 {separable_count}개, 일반 {regular_count}개"
        )

        return results

    def initialize_separable_detection(self, user_words_file=None):
        """분리형 숙어 감지 시스템 초기화"""

        print(f"🔧 분리형 숙어 감지 시스템 초기화...")

        # SeparableIdiomDetector 인스턴스 생성
        self.separable_detector = SeparableIdiomDetector(client, verbose=self.verbose)

        # 기존 분석 결과 로드 시도
        cache_file = "separable_analysis.json"
        if not self.separable_detector.load_separable_analysis(cache_file):
            # 캐시가 없으면 새로 분석
            if hasattr(self, "user_idioms") and self.user_idioms:
                print(f"🤖 사용자 숙어 분리형 분석 시작...")
                separable_analysis = (
                    self.separable_detector.analyze_user_idioms_with_gpt(
                        self.user_idioms
                    )
                )
                self.separable_detector.build_separable_patterns(separable_analysis)

                # 분석 결과 저장
                self.separable_detector.save_separable_analysis(cache_file)
            else:
                print("⚠️ 사용자 숙어가 없어 분리형 분석을 건너뜁니다")

        # extractor와 연동
        self.user_separable_idioms = self.separable_detector.user_separable_idioms

        print(
            f"✅ 분리형 감지 시스템 초기화 완료: {len(self.user_separable_idioms)}개 분리형 숙어"
        )

    # 🔥 새로운 사용자 단어 로딩 함수 (숙어 감지 포함)
    def _load_user_words_with_idiom_detection(self, user_words_file):
        """사용자 단어 파일에서 숙어와 단일 단어를 구분하여 로딩"""
        try:
            if user_words_file.endswith(".csv"):
                # 🔥 여러 인코딩 시도
                encodings = ["utf-8", "cp949", "euc-kr", "latin1", "utf-8-sig"]
                user_df = None

                for encoding in encodings:
                    try:
                        user_df = pd.read_csv(user_words_file, encoding=encoding)
                        print(f"   ✅ 인코딩 성공: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue

                if user_df is None:
                    print(f"   ❌ 모든 인코딩 시도 실패")
                    return

            elif user_words_file.endswith((".xlsx", ".xls")):
                user_df = pd.read_excel(user_words_file)
            else:
                print(f"   ⚠️ 지원하지 않는 파일 형식")
                return

            if not user_df.empty and len(user_df.columns) > 0:
                # 첫 번째 컬럼의 단어들 추출
                user_words = (
                    user_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                )

                # 🔥 숙어와 단일 단어 분리
                idiom_count = 0
                single_word_count = 0

                for word in user_words:
                    word_clean = word.lower().strip()
                    self.user_words.add(word_clean)

                    # 띄어쓰기가 있으면 숙어로 분류
                    if " " in word_clean and len(word_clean.split()) >= 2:
                        self.user_idioms.add(word_clean)
                        idiom_count += 1
                    else:
                        self.user_single_words.add(word_clean)
                        single_word_count += 1

                print(f"   ✅ 총 {len(user_words)}개 단어 로드 완료")
                print(f"   📋 숙어: {idiom_count}개")
                print(f"   📋 단일 단어: {single_word_count}개")

                # 🔥 사용자 숙어 샘플 출력
                if self.user_idioms:
                    sample_idioms = list(self.user_idioms)[:5]
                    print(f"   📝 사용자 숙어 예시: {sample_idioms}")

            else:
                print(f"   ⚠️ 파일이 비어있거나 읽을 수 없음")

            # 사용자 단어 로드 완료 후 GPT 필터 재초기화
            if hasattr(self, "gpt_filter") and self.gpt_filter:
                self.gpt_filter.user_words = self.user_single_words
                print(
                    f"   🔄 GPT 필터 사용자 단어 업데이트: {len(self.user_single_words)}개"
                )

        except Exception as e:
            print(f"   ❌ 사용자 단어 파일 로드 실패: {e}")

    def _load_easy_words(self):
        """쉬운 단어 목록 로드 (Excel 파일 지원 추가)"""
        try:
            # 1. 먼저 pickle 캐시 확인
            easy_words_cache = self.settings[
                "EASY_WORDS_CACHE"
            ]  # "elementary_words.pkl"
            if os.path.exists(easy_words_cache):
                with open(easy_words_cache, "rb") as f:
                    easy_words = pickle.load(f)
                print(f"✅ 쉬운 단어 목록 {len(easy_words)}개 캐시에서 로드 완료")
                return easy_words

            # 2. Excel 파일 확인
            excel_files = [
                "easy_words.xlsx",
                "elementary_words.xlsx",
                "basic_words.xlsx",
            ]
            for excel_file in excel_files:
                if os.path.exists(excel_file):
                    print(f"📊 Excel 파일에서 쉬운 단어 로딩: {excel_file}")
                    try:
                        df = pd.read_excel(excel_file)
                        # 첫 번째 컬럼의 단어들 추출
                        words_column = df.columns[0]
                        easy_words = set(
                            df[words_column]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )

                        # pickle 캐시로 저장 (다음번에 빠르게 로드하기 위해)
                        try:
                            with open(easy_words_cache, "wb") as f:
                                pickle.dump(easy_words, f)
                            print(f"✅ 캐시 저장 완료: {easy_words_cache}")
                        except Exception as e:
                            print(f"⚠️ 캐시 저장 실패: {e}")

                        print(f"✅ Excel에서 쉬운 단어 {len(easy_words)}개 로드 완료")
                        return easy_words

                    except Exception as e:
                        print(f"⚠️ {excel_file} 로드 실패: {e}")
                        continue

            # 3. CSV 파일도 확인
            csv_files = ["easy_words.csv", "elementary_words.csv", "basic_words.csv"]
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    print(f"📊 CSV 파일에서 쉬운 단어 로딩: {csv_file}")
                    try:
                        df = pd.read_csv(csv_file, encoding="utf-8")
                        words_column = df.columns[0]
                        easy_words = set(
                            df[words_column]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )

                        print(f"✅ CSV에서 쉬운 단어 {len(easy_words)}개 로드 완료")
                        return easy_words

                    except Exception as e:
                        print(f"⚠️ {csv_file} 로드 실패: {e}")
                        continue

        except Exception as e:
            print(f"⚠️ 쉬운 단어 목록 로드 실패: {e}")

        # 4. 모든 파일 로드 실패 시 기본 단어들 사용
        print("📚 기본 단어 세트 사용")
        return set(stopwords.words("english")).union(
            BASIC_VERBS,
            BASIC_ADJECTIVES,
            BASIC_NOUNS,
            {
                "the",
                "a",
                "an",
                "this",
                "that",
                "these",
                "those",
                "good",
                "bad",
                "big",
                "small",
                "new",
                "old",
                "young",
                "make",
                "take",
                "get",
                "go",
                "come",
                "see",
                "know",
            },
        )

    def _build_frequency_from_csv(self, csv_file):
        """CSV 파일에서 빈도 데이터 구축 (개선된 버전)"""
        try:
            print(f"📊 빈도 분석 시작: {csv_file}")

            # 🔥 여러 인코딩 시도
            encodings = ["utf-8", "cp949", "euc-kr", "latin1", "utf-8-sig"]
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"   ✅ 파일 읽기 성공: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                print(f"   ❌ 파일 읽기 실패: 모든 인코딩 시도 실패")
                return {}

            print(f"   📋 데이터프레임 크기: {len(df)} rows, {len(df.columns)} columns")
            print(f"   📋 컬럼명: {list(df.columns)}")

            # 텍스트 컬럼 찾기
            text_columns = [
                "content",
                "지문",
                "text",
                "텍스트",
                "본문",
                "Content",
                "TEXT",
            ]
            found_column = None

            for col in text_columns:
                if col in df.columns:
                    found_column = col
                    break

            if not found_column:
                print(
                    f"   ⚠️ 텍스트 컬럼을 찾을 수 없음. 사용 가능한 컬럼: {list(df.columns)}"
                )
                # 첫 번째 컬럼을 텍스트 컬럼으로 사용
                if len(df.columns) > 0:
                    found_column = df.columns[0]
                    print(f"   📝 첫 번째 컬럼 사용: {found_column}")
                else:
                    return {}

            # 텍스트 추출
            texts = df[found_column].dropna().astype(str).tolist()
            print(f"   📝 추출된 텍스트: {len(texts)}개")

            if not texts:
                print(f"   ⚠️ 추출된 텍스트가 없음")
                return {}

            # 빈도 분석
            freq_result = self._calculate_word_frequencies(texts)
            print(f"   ✅ 빈도 분석 완료: {len(freq_result)}개 단어")

            return freq_result

        except Exception as e:
            print(f"❌ 빈도 분석 실패: {e}")
            import traceback

            traceback.print_exc()
        return {}

    def _calculate_word_frequencies(self, texts):
        """텍스트에서 단어 빈도 계산"""
        word_counts = Counter()
        for text in texts:
            doc = nlp(text)
            for token in doc:
                if (
                    len(token.text) >= 3
                    and token.is_alpha
                    and not token.is_stop
                    and token.pos_ not in ["PUNCT", "SPACE", "SYM"]
                ):
                    lemma = token.lemma_.lower()
                    if lemma != "-PRON-":
                        word_counts[lemma] += 1

        # 빈도순으로 정렬하여 등급 부여
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        freq_tiers = {}
        total_words = len(sorted_words)

        for rank, (word, count) in enumerate(sorted_words, 1):
            if rank <= total_words * 0.05:
                tier = 1.0
            elif rank <= total_words * 0.2:
                tier = 2.0
            elif rank <= total_words * 0.5:
                tier = 3.0
            else:
                tier = 4.0 + (rank - total_words * 0.5) / (total_words * 0.5)
            freq_tiers[word] = tier

        print(f"✅ 단어 빈도 분석 완료: {len(freq_tiers)}개 단어")
        return freq_tiers

    def save_cache_to_file(self, cache_file=None):
        """GPT 캐시 저장"""
        if not cache_file:
            cache_file = self.settings["CACHE_FILE"]

        serializable_cache = {str(k): v for k, v in self.gpt_cache.items()}
        cache_data = {
            "cache": serializable_cache,
            "token_usage": self.gpt_token_usage,
            "call_count": self.gpt_call_count,
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"✅ GPT 캐시 저장 완료: {len(self.gpt_cache)}개 항목")
        except Exception as e:
            print(f"⚠️ GPT 캐시 저장 실패: {e}")

    def load_cache_from_file(self, cache_file=None):
        """GPT 캐시 로드"""
        if not cache_file:
            cache_file = self.settings["CACHE_FILE"]

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                if isinstance(cache_data, dict) and "cache" in cache_data:
                    serialized_cache = cache_data["cache"]
                    if "token_usage" in cache_data:
                        self.gpt_token_usage = cache_data["token_usage"]
                    if "call_count" in cache_data:
                        self.gpt_call_count = cache_data["call_count"]
                else:
                    serialized_cache = cache_data

                for k, v in serialized_cache.items():
                    try:
                        self.gpt_cache[k] = v
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️ 캐시 항목 변환 실패: {k} - {e}")

                print(f"✅ GPT 캐시 로드 완료: {len(self.gpt_cache)}개 항목")
            except Exception as e:
                print(f"⚠️ GPT 캐시 로드 실패: {e}")
                self.gpt_cache = {}

    # 🔥 통합된 의미 생성 함수
    def enhanced_korean_definition(
        self, word, sentence, is_phrase=False, pos_hint=None
    ):
        """문맥을 고려한 향상된 의미 생성"""
        if not self.settings["USE_INTEGRATED_CONTEXTUAL"]:
            return self._legacy_korean_definition(word, sentence, is_phrase, pos_hint)

        try:
            result, updated_cache, updated_call_count, updated_token_usage = (
                integrated_get_best_korean_definition(
                    word=word,
                    phrase_db=self.phrase_db,
                    is_phrase=is_phrase,
                    max_tokens=self.MAX_TOKENS,
                    client=client,
                    gpt_cache=self.gpt_cache,
                    gpt_call_count=self.gpt_call_count,
                    GPT_CALL_LIMIT=self.GPT_CALL_LIMIT,
                    token_usage=self.gpt_token_usage,
                    custom_prompt=None,
                    sentence=sentence,  # 🔥
                )
            )

            # 상태 업데이트
            self.gpt_cache = updated_cache
            self.gpt_call_count = updated_call_count
            self.gpt_token_usage = updated_token_usage

            return self._clean_korean_definition(word, result)

        except Exception as e:
            print(f"   ❌ integrated_get_best_korean_definition 실패: {e}")
            return word

    def integrated_comprehensive_analysis(self, word, context="", pos=""):
        """Single GPT call for complete word analysis"""

        cache_key = f"comprehensive:{word.lower()}:{context[:50]}"
        if cache_key in self.gpt_cache:
            return self.gpt_cache[cache_key]

        # Check if it's an idiom/phrase first
        is_idiom = " " in word or "-" in word or "~" in word

        comprehensive_prompt = f"""
Analyze the English word/phrase: "{word}"
Context: "{context}"
Part of Speech: {pos}

Provide comprehensive analysis in JSON format:

1. Basic Information:
   - Korean meaning (concise, 2-5 words)
   - Confirmed part of speech
   - Is this the base/root form?
   - 2-3 synonyms
   - 1-2 antonyms

2. Difficulty Analysis:
   - Difficulty score (0.5-10 scale)
     * 0.5-2: Basic elementary words (a, the, cat, red)
     * 3-4: Elementary level (big, good, come, go)
     * 5-6: Intermediate level (difficult, because, important)
     * 7-8: Advanced level (significant, comprehensive)
     * 9-10: Expert level (sophisticated, paradigm)
   - Level category (elementary/advanced)
   - Recommendation (keep/remove)
   - Detailed reasoning

3. Quality Assessment:
   - Is this a proper noun? (person/place/brand names)
   - Is this a separable phrasal verb? (needs ~ notation)
   - Tilde consistency check
   - Quality score (0-100)
   - Identified issues

4. Confidence Evaluation:
   - Analysis certainty (1-10)
   - Needs secondary verification?
   - Risk factors for re-evaluation

Special considerations:
- Korean high school student perspective
- Educational curriculum context
- Separable phrasal verb patterns (pick up → pick ~ up)
- Collocation importance (make effort, take care)

Response format:
{{
    "basic_info": {{
        "korean_meaning": "Korean translation",
        "pos": "confirmed part of speech",
        "is_base_form": true/false,
        "suggested_base": "base form if not base",
        "synonyms": ["synonym1", "synonym2"],
        "antonyms": ["antonym1"]
    }},
    "difficulty": {{
        "score": 7.5,
        "level": "advanced",
        "recommendation": "keep",
        "reasoning": "detailed explanation",
        "educational_value": "high/medium/low",
        "is_collocation_part": true/false
    }},
    "quality": {{
        "is_proper_noun": false,
        "proper_noun_type": "none/person/place/brand",
        "is_separable": false,
        "needs_tilde": false,
        "separable_pattern": "none/pick ~ up format",
        "quality_score": 85,
        "issues": ["list of quality issues"]
    }},
    "confidence": {{
        "certainty": 8,
        "needs_verification": false,
        "risk_factors": ["factors requiring re-evaluation"],
        "verification_priority": "none/low/medium/high"
    }}
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Cost optimization
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert English educator specializing in Korean curriculum. Provide accurate, comprehensive analysis for vocabulary assessment.",
                    },
                    {"role": "user", "content": comprehensive_prompt},
                ],
                max_tokens=800,  # Increased for comprehensive analysis
                temperature=0.1,
            )

            # Parse and cache result
            content = response.choices[0].message.content.strip()
            result = self._parse_comprehensive_result(content, word, is_idiom)

            # Update tracking
            self.gpt_cache[cache_key] = result
            self.gpt_call_count += 1

            # Update token usage
            if hasattr(response, "usage"):
                usage = response.usage
                self.gpt_token_usage["prompt_tokens"] += usage.prompt_tokens
                self.gpt_token_usage["completion_tokens"] += usage.completion_tokens
                self.gpt_token_usage["total_tokens"] += usage.total_tokens

            if self.verbose:
                score = result.get("difficulty", {}).get("score", "unknown")
                level = result.get("difficulty", {}).get("level", "unknown")
                print(f"      🔍 Comprehensive: {word} → {level} ({score}pts)")

            return result

        except Exception as e:
            if self.verbose:
                print(f"   ❌ Comprehensive analysis failed ({word}): {e}")
            return self._get_fallback_comprehensive_result(word, is_idiom)

    def _parse_comprehensive_result(self, content, word, is_idiom):
        """Parse comprehensive analysis JSON result"""
        try:
            # Extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()

            result = json.loads(content)

            # Validate and enhance result
            result = self._validate_comprehensive_result(result, word, is_idiom)

            return result

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ JSON parsing failed: {e}")
            return self._get_fallback_comprehensive_result(word, is_idiom)

    def _validate_comprehensive_result(self, result, word, is_idiom):
        """Validate and enhance comprehensive analysis result"""

        # Ensure all required sections exist
        if "basic_info" not in result:
            result["basic_info"] = {}
        if "difficulty" not in result:
            result["difficulty"] = {}
        if "quality" not in result:
            result["quality"] = {}
        if "confidence" not in result:
            result["confidence"] = {}

        # Validate basic_info
        basic = result["basic_info"]
        if "korean_meaning" not in basic or not basic["korean_meaning"]:
            basic["korean_meaning"] = word  # Fallback
        if "is_base_form" not in basic:
            basic["is_base_form"] = True
        if "synonyms" not in basic:
            basic["synonyms"] = []
        if "antonyms" not in basic:
            basic["antonyms"] = []

        # Validate difficulty
        difficulty = result["difficulty"]
        try:
            score = float(difficulty.get("score", 5.0))
            difficulty["score"] = max(0.5, min(10.0, score))  # Ensure 0.5-10 range
        except (ValueError, TypeError):
            difficulty["score"] = 5.0

        if difficulty.get("level") not in ["elementary", "advanced"]:
            difficulty["level"] = (
                "advanced" if difficulty["score"] >= 6 else "elementary"
            )

        if difficulty.get("recommendation") not in ["keep", "remove"]:
            difficulty["recommendation"] = (
                "keep" if difficulty["score"] >= 5 else "remove"
            )

        # Special handling for idioms
        if is_idiom:
            difficulty["score"] = max(
                7.0, difficulty["score"]
            )  # Idioms are at least 7.0
            difficulty["level"] = "advanced"
            difficulty["recommendation"] = "keep"
            difficulty["educational_value"] = "high"

        # Validate quality
        quality = result["quality"]
        if "quality_score" not in quality:
            quality["quality_score"] = 80  # Default good quality
        if "is_proper_noun" not in quality:
            quality["is_proper_noun"] = False
        if "is_separable" not in quality:
            quality["is_separable"] = False
        if "issues" not in quality:
            quality["issues"] = []

        # Validate confidence
        confidence = result["confidence"]
        try:
            certainty = int(confidence.get("certainty", 8))
            confidence["certainty"] = max(1, min(10, certainty))
        except (ValueError, TypeError):
            confidence["certainty"] = 8

        if "needs_verification" not in confidence:
            confidence["needs_verification"] = confidence["certainty"] < 7
        if "risk_factors" not in confidence:
            confidence["risk_factors"] = []
        if "verification_priority" not in confidence:
            confidence["verification_priority"] = "none"

        # Add user DB information
        word_lower = word.lower()
        result["user_db_info"] = {
            "in_user_words": word_lower in self.user_single_words,
            "in_user_idioms": word_lower in self.user_idioms,
            "user_priority": word_lower in self.user_words,
        }

        return result

    def _get_fallback_comprehensive_result(self, word, is_idiom):
        """Fallback result when GPT analysis fails"""

        word_lower = word.lower()

        # Basic heuristics
        if is_idiom:
            score, level, recommendation = 8.0, "advanced", "keep"
        elif len(word) <= 4:
            score, level, recommendation = 3.0, "elementary", "remove"
        elif len(word) <= 6:
            score, level, recommendation = 5.0, "elementary", "remove"
        else:
            score, level, recommendation = 7.0, "advanced", "keep"

        return {
            "basic_info": {
                "korean_meaning": word,
                "pos": "unknown",
                "is_base_form": True,
                "synonyms": [],
                "antonyms": [],
            },
            "difficulty": {
                "score": score,
                "level": level,
                "recommendation": recommendation,
                "reasoning": "Fallback analysis due to GPT failure",
                "educational_value": "high" if is_idiom else "medium",
            },
            "quality": {
                "is_proper_noun": False,
                "is_separable": False,
                "needs_tilde": False,
                "quality_score": 70,
                "issues": ["GPT analysis failed"],
            },
            "confidence": {
                "certainty": 5,
                "needs_verification": True,
                "risk_factors": ["fallback_analysis"],
                "verification_priority": "high",
            },
            "user_db_info": {
                "in_user_words": word_lower in self.user_single_words,
                "in_user_idioms": word_lower in self.user_idioms,
                "user_priority": word_lower in self.user_words,
            },
            "fallback": True,
        }

    def _should_include_word_comprehensive(self, word, comprehensive_result):
        """Determine if word should be included based on comprehensive analysis"""

        word_lower = word.lower()

        # 1. User DB priority
        if comprehensive_result["user_db_info"]["user_priority"]:
            return True, "user_db_priority"

        # 2. Idioms/phrases always keep
        if " " in word or "-" in word or "~" in word:
            return True, "idiom_pattern"

        # 3. GPT recommendation
        gpt_rec = comprehensive_result["difficulty"]["recommendation"]
        if gpt_rec == "keep":
            return (
                True,
                f"gpt_recommended_{comprehensive_result['difficulty']['score']}pts",
            )

        # 4. Quality issues
        quality_score = comprehensive_result["quality"]["quality_score"]
        if quality_score < 50:
            return False, f"poor_quality_{quality_score}pts"

        # 5. Proper noun check
        if comprehensive_result["quality"]["is_proper_noun"]:
            return False, "proper_noun_excluded"

        # 6. Final decision based on difficulty
        difficulty_score = comprehensive_result["difficulty"]["score"]
        if difficulty_score >= 5.5:
            return True, f"sufficient_difficulty_{difficulty_score}pts"

        return False, f"insufficient_difficulty_{difficulty_score}pts"

    def _parse_comprehensive_result(self, content, word, is_idiom):
        """Parse comprehensive analysis JSON result"""
        try:
            # Extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()

            result = json.loads(content)

            # Validate and enhance result
            result = self._validate_comprehensive_result(result, word, is_idiom)

            return result

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ JSON parsing failed: {e}")
            return self._get_fallback_comprehensive_result(word, is_idiom)

    def _validate_comprehensive_result(self, result, word, is_idiom):
        """Validate and enhance comprehensive analysis result"""

        # Ensure all required sections exist
        if "basic_info" not in result:
            result["basic_info"] = {}
        if "difficulty" not in result:
            result["difficulty"] = {}
        if "quality" not in result:
            result["quality"] = {}
        if "confidence" not in result:
            result["confidence"] = {}

        # Validate basic_info
        basic = result["basic_info"]
        if "korean_meaning" not in basic or not basic["korean_meaning"]:
            basic["korean_meaning"] = word  # Fallback
        if "is_base_form" not in basic:
            basic["is_base_form"] = True
        if "synonyms" not in basic:
            basic["synonyms"] = []
        if "antonyms" not in basic:
            basic["antonyms"] = []

        # Validate difficulty
        difficulty = result["difficulty"]
        try:
            score = float(difficulty.get("score", 5.0))
            difficulty["score"] = max(0.5, min(10.0, score))  # Ensure 0.5-10 range
        except (ValueError, TypeError):
            difficulty["score"] = 5.0

        if difficulty.get("level") not in ["elementary", "advanced"]:
            difficulty["level"] = (
                "advanced" if difficulty["score"] >= 6 else "elementary"
            )

        if difficulty.get("recommendation") not in ["keep", "remove"]:
            difficulty["recommendation"] = (
                "keep" if difficulty["score"] >= 5 else "remove"
            )

        # Special handling for idioms
        if is_idiom:
            difficulty["score"] = max(
                7.0, difficulty["score"]
            )  # Idioms are at least 7.0
            difficulty["level"] = "advanced"
            difficulty["recommendation"] = "keep"
            difficulty["educational_value"] = "high"

        # Validate quality
        quality = result["quality"]
        if "quality_score" not in quality:
            quality["quality_score"] = 80  # Default good quality
        if "is_proper_noun" not in quality:
            quality["is_proper_noun"] = False
        if "is_separable" not in quality:
            quality["is_separable"] = False
        if "issues" not in quality:
            quality["issues"] = []

        # Validate confidence
        confidence = result["confidence"]
        try:
            certainty = int(confidence.get("certainty", 8))
            confidence["certainty"] = max(1, min(10, certainty))
        except (ValueError, TypeError):
            confidence["certainty"] = 8

        if "needs_verification" not in confidence:
            confidence["needs_verification"] = confidence["certainty"] < 7
        if "risk_factors" not in confidence:
            confidence["risk_factors"] = []
        if "verification_priority" not in confidence:
            confidence["verification_priority"] = "none"

        # Add user DB information
        word_lower = word.lower()
        result["user_db_info"] = {
            "in_user_words": word_lower in self.user_single_words,
            "in_user_idioms": word_lower in self.user_idioms,
            "user_priority": word_lower in self.user_words,
        }

        return result

    def _get_fallback_comprehensive_result(self, word, is_idiom):
        """Fallback result when GPT analysis fails"""

        word_lower = word.lower()

        # Basic heuristics
        if is_idiom:
            score, level, recommendation = 8.0, "advanced", "keep"
        elif len(word) <= 4:
            score, level, recommendation = 3.0, "elementary", "remove"
        elif len(word) <= 6:
            score, level, recommendation = 5.0, "elementary", "remove"
        else:
            score, level, recommendation = 7.0, "advanced", "keep"

        return {
            "basic_info": {
                "korean_meaning": word,
                "pos": "unknown",
                "is_base_form": True,
                "synonyms": [],
                "antonyms": [],
            },
            "difficulty": {
                "score": score,
                "level": level,
                "recommendation": recommendation,
                "reasoning": "Fallback analysis due to GPT failure",
                "educational_value": "high" if is_idiom else "medium",
            },
            "quality": {
                "is_proper_noun": False,
                "is_separable": False,
                "needs_tilde": False,
                "quality_score": 70,
                "issues": ["GPT analysis failed"],
            },
            "confidence": {
                "certainty": 5,
                "needs_verification": True,
                "risk_factors": ["fallback_analysis"],
                "verification_priority": "high",
            },
            "user_db_info": {
                "in_user_words": word_lower in self.user_single_words,
                "in_user_idioms": word_lower in self.user_idioms,
                "user_priority": word_lower in self.user_words,
            },
            "fallback": True,
        }

    def _clean_korean_definition(self, word, text):
        """한글 뜻 정리 함수"""
        if not isinstance(text, str):
            return ""

        text = text.strip("\"'").strip()
        text = (
            text.replace('"', "")
            .replace("'", "")
            .replace(
                """, "")
            .replace(""",
                "",
            )
            .replace("'", "")
            .replace("'", "")
        )

        # ✅ 숙어/패턴인지 확인
        if " A " in word or " B " in word or " " in word.strip():
            return text  # 영어 표현 유지

        # ✅ 일반 단어인 경우 영어 제거
        text = re.sub(r"[a-zA-Z]{3,}", "", text)  # 3글자 이상 영어 단어 제거
        return text.strip()

    def _legacy_korean_definition(self, word, sentence, is_phrase, pos_hint):
        """기존 방식의 의미 생성 (호환성용)"""
        basic_definitions = {
            "the": "그",
            "a": "하나의",
            "an": "하나의",
            "and": "그리고",
            "or": "또는",
            "but": "하지만",
            "in": "~안에",
            "on": "~위에",
        }

        if word.lower() in basic_definitions:
            return basic_definitions[word.lower()]

        if is_phrase:
            return f"{word}의 숙어적 의미"
        else:
            return f"{word}의 의미"

    # 🔥 통합된 난이도 분석
    def enhanced_difficulty_analysis(self, word, context=""):
        """통합된 난이도 분석"""
        if not self.settings["USE_INTEGRATED_DIFFICULTY"]:
            return self._legacy_difficulty_analysis(word)

        try:
            # 🔥 통합된 vocab_difficulty 함수들 사용
            definition, synonyms, antonyms = integrated_extract_info(word)
            difficulty_score = integrated_get_word_difficulty_score(word, nlp)

            # 추가 분석
            factors = []

            # 길이 기반 난이도
            length_score = min(len(word) / 10, 1.0)
            factors.append(f"길이: {length_score:.2f}")

            # WordNet 복잡도
            synsets = wordnet.synsets(word)
            synset_score = min(len(synsets) / 5, 1.0)
            factors.append(f"의미수: {synset_score:.2f}")

            # 음성적 복잡도
            phonetic_score = integrated_calculate_phonetic_complexity(word)
            factors.append(f"음성: {phonetic_score:.2f}")

            # 최종 난이도 레벨 결정
            if difficulty_score < 0.3:
                level = "easy"
            elif difficulty_score < 0.6:
                level = "medium"
            elif difficulty_score < 0.8:
                level = "hard"
            else:
                level = "very_hard"

            result = {
                "word": word,
                "difficulty_score": difficulty_score,
                "level": level,
                "factors": factors,
                "definition": definition,
                "synonyms": synonyms,
                "antonyms": antonyms,
                "synset_count": len(synsets),
                "phonetic_complexity": phonetic_score,
            }

            return result

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 통합 난이도 분석 실패 ({word}): {e}")
            return self._legacy_difficulty_analysis(word)

    def _should_include_word_comprehensive(self, word, comprehensive_result):
        """Determine if word should be included based on comprehensive analysis"""

        word_lower = word.lower()

        # 1. User DB priority
        if comprehensive_result["user_db_info"]["user_priority"]:
            return True, "user_db_priority"

        # 2. Idioms/phrases always keep
        if " " in word or "-" in word or "~" in word:
            return True, "idiom_pattern"

        # 3. GPT recommendation
        gpt_rec = comprehensive_result["difficulty"]["recommendation"]
        if gpt_rec == "keep":
            return (
                True,
                f"gpt_recommended_{comprehensive_result['difficulty']['score']}pts",
            )

        # 4. Quality issues
        quality_score = comprehensive_result["quality"]["quality_score"]
        if quality_score < 50:
            return False, f"poor_quality_{quality_score}pts"

        # 5. Proper noun check
        if comprehensive_result["quality"]["is_proper_noun"]:
            return False, "proper_noun_excluded"

        # 6. Final decision based on difficulty
        difficulty_score = comprehensive_result["difficulty"]["score"]
        if difficulty_score >= 6.0:
            return True, f"sufficient_difficulty_{difficulty_score}pts"

        return False, f"insufficient_difficulty_{difficulty_score}pts"

    def _legacy_difficulty_analysis(self, word):
        """기존 방식의 난이도 분석 (호환성용)"""
        return {
            "word": word,
            "difficulty_score": 0.5,
            "level": "medium",
            "factors": ["기본분석"],
            "definition": "",
            "synonyms": [],
            "antonyms": [],
        }

    # 🔥 새로운 사용자 DB 숙어 우선 추출
    def extract_user_db_idioms(self, text):
        """사용자 DB에서 숙어 우선 추출"""
        results = []
        text_str = self._force_extract_text(text)
        text_lower = text_str.lower()
        found_positions = set()

        print(f"   🔍 사용자 DB 숙어 매칭 검사: {len(self.user_idioms)}개")

        if not self.user_idioms:
            print(f"   ⚠️ 사용자 DB에 숙어 없음")
            return results

        # 🔥 길이순 정렬 (긴 숙어부터 매칭하여 중복 방지)
        sorted_user_idioms = sorted(self.user_idioms, key=len, reverse=True)

        for idiom in sorted_user_idioms:
            # 정확한 매칭 (단어 경계 고려)
            pattern = r"\b" + re.escape(idiom) + r"\b"
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)

            for match in matches:
                start, end = match.span()

                # 위치 중복 확인
                if any(abs(start - pos[0]) < 5 for pos in found_positions):
                    continue

                context = self._get_sentence_context(text_str, start, end)

                # 원본 텍스트에서 실제 표현 추출
                original_text = text_str[start:end]

                # 한글 의미 생성
                meaning = self.enhanced_korean_definition(
                    idiom, context, is_phrase=True
                )

                results.append(
                    {
                        "original": original_text,
                        "base_form": idiom,
                        "meaning": meaning,
                        "context": context,
                        "type": "user_db_idiom",  # 🔥 사용자 DB 숙어 표시
                        "is_separated": False,
                        "confidence": 0.95,  # 사용자 DB는 높은 신뢰도
                        "user_db_match": True,
                        "match_type": "사용자DB숙어",
                    }
                )
                found_positions.add((start, end))

        return results

    # 🔥 개선된 숙어 추출 (사용자 DB 우선 + 고급 패턴 분석)
    def extract_advanced_idioms(self, text):
        """개선된 숙어 추출 - 사용자 DB 우선 + 분리형/문법패턴 분석"""
        results = []
        text_str = self._force_extract_text(text)
        found_positions = set()

        try:
            # 🔥 1. 사용자 DB 숙어 우선 검사 (최우선)
            user_idioms = self.extract_user_db_idioms(text)
            results.extend(user_idioms)

            # 사용자 DB 숙어 위치 기록
            for idiom in user_idioms:
                start = text_str.lower().find(idiom["base_form"].lower())
                if start != -1:
                    end = start + len(idiom["base_form"])
                    found_positions.add((start, end))

            # 🔥 2. 문법 패턴 숙어 검사 (V-ing, N V 등)
            print(f"   🔍 문법 패턴 분석 중...")
            grammar_patterns = self.idiom_checker.analyze_grammar_pattern(text_str)

            for pattern in grammar_patterns:
                # 위치 중복 확인
                if any(abs(pattern["start"] - pos[0]) < 10 for pos in found_positions):
                    continue

                context = self._get_sentence_context(
                    text_str, pattern["start"], pattern["end"]
                )
                meaning = self.enhanced_korean_definition(
                    pattern["display_form"], context, is_phrase=True
                )

                results.append(
                    {
                        "original": pattern["original"],
                        "base_form": pattern[
                            "display_form"
                        ],  # 🔥 패턴 표시 (spend time V-ing)
                        "meaning": meaning,
                        "context": context,
                        "type": "grammar_pattern",
                        "is_separated": pattern["is_separated"],
                        "confidence": 0.9,
                        "user_db_match": False,
                        "match_type": "문법패턴",
                    }
                )
                found_positions.add((pattern["start"], pattern["end"]))
                print(
                    f"      ✅ 문법 패턴: '{pattern['original']}' → {pattern['display_form']}"
                )

            # 🔥 3. 참조 DB 숙어 (사용자 DB 다음 우선순위)
            print(f"   🔍 참조 숙어 DB에서 매칭 검사: {len(self.reference_idioms)}개")
            for idiom in self.reference_idioms:
                if idiom.lower() in text_str.lower():
                    start = text_str.lower().find(idiom.lower())
                    end = start + len(idiom)

                    # 위치 중복 확인
                    if any(abs(start - pos[0]) < 10 for pos in found_positions):
                        continue

                    context = self._get_sentence_context(text_str, start, end)
                    meaning = self.enhanced_korean_definition(
                        idiom, context, is_phrase=True
                    )

                    results.append(
                        {
                            "original": idiom,
                            "base_form": idiom,
                            "meaning": meaning,
                            "context": context,
                            "type": "reference_idiom_db",
                            "is_separated": False,
                            "confidence": 0.85,
                            "user_db_match": False,
                            "match_type": "참조DB",
                        }
                    )
                    found_positions.add((start, end))

            # 🔥 4. SpaCy 기반 고급 구동사 추출
            spacy_results = self._extract_advanced_phrasal_verbs(
                text_str, found_positions
            )
            results.extend(spacy_results)

            # 🔥 5. 간단한 패턴 매칭
            pattern_results = self._extract_simple_patterns(text_str, found_positions)
            results.extend(pattern_results)

            # 신뢰도 및 사용자 DB 우선순위로 정렬
            results.sort(
                key=lambda x: (-x.get("user_db_match", False), -x.get("confidence", 0))
            )

        except Exception as e:
            if self.verbose:
                print(f"숙어 추출 오류: {e}")

        return results

    def _extract_advanced_phrasal_verbs(self, text, found_positions):
        """고급 구동사 추출 - 연속형/분리형/패턴 구분"""
        results = []
        try:
            doc = nlp(text)

            # 의존성 파싱을 이용한 구동사 탐지
            for token in doc:
                if token.pos_ == "VERB":
                    # 동사의 모든 자식 토큰들 확인
                    particles = []

                    for child in token.children:
                        # particle (prt) 의존성을 가진 토큰들 찾기
                        if child.dep_ == "prt":
                            particles.append(child)

                    # 구동사가 발견되면
                    if particles:
                        for particle in particles:
                            # 🔥 고급 패턴 분석
                            pattern_analysis = (
                                self.idiom_checker.analyze_phrasal_verb_pattern(
                                    text, token, particle
                                )
                            )

                            # 위치 계산
                            if pattern_analysis["is_separated"]:
                                # 분리형인 경우 전체 범위 계산
                                start = min(token.idx, particle.idx)
                                end = max(
                                    token.idx + len(token.text),
                                    particle.idx + len(particle.text),
                                )
                                # 중간에 있는 단어들도 포함
                                for i in range(
                                    min(token.i, particle.i) + 1,
                                    max(token.i, particle.i),
                                ):
                                    if i < len(doc):
                                        end = max(end, doc[i].idx + len(doc[i].text))
                                original_text = text[start:end].strip()
                            else:
                                # 연속형인 경우
                                start = min(token.idx, particle.idx)
                                end = max(
                                    token.idx + len(token.text),
                                    particle.idx + len(particle.text),
                                )
                                original_text = text[start:end].strip()

                            # 위치 중복 확인
                            if any(abs(start - pos[0]) < 5 for pos in found_positions):
                                continue

                            context = self._get_sentence_context(text, start, end)

                            # 🔥 표시 형태 결정
                            display_form = pattern_analysis["display_form"]
                            base_form = pattern_analysis["base_form"]

                            meaning = self.enhanced_korean_definition(
                                base_form, context, is_phrase=True
                            )

                            results.append(
                                {
                                    "original": original_text,
                                    "base_form": display_form,  # 🔥 패턴에 따른 표시 (pick up vs pick ~ up)
                                    "meaning": meaning,
                                    "context": context,
                                    "type": f"advanced_phrasal_{pattern_analysis['pattern_type']}",
                                    "is_separated": pattern_analysis["is_separated"],
                                    "confidence": (
                                        0.9 if pattern_analysis["is_separated"] else 0.8
                                    ),
                                    "user_db_match": False,
                                    "match_type": f"고급구동사({pattern_analysis['pattern_type']})",
                                    "pattern_info": pattern_analysis,  # 🔥 패턴 상세 정보
                                }
                            )
                            found_positions.add((start, end))

                            pattern_desc = {
                                "optional_separable": "연속형가능",
                                "mandatory_separable": "분리필수",
                                "actually_separated": "실제분리",
                                "continuous": "연속형",
                            }.get(
                                pattern_analysis["pattern_type"],
                                pattern_analysis["pattern_type"],
                            )

                            print(
                                f"      ✅ 고급 구동사 ({pattern_desc}): '{original_text}' → {display_form}"
                            )

        except Exception as e:
            if self.verbose:
                print(f"고급 구동사 추출 오류: {e}")

        return results

    def _extract_simple_patterns(self, text, found_positions):
        """간단한 패턴 매칭"""
        results = []

        # 알려진 숙어 패턴들
        simple_patterns = [
            r"\bas\s+\w+\s+as\b",  # as ... as
            r"\bin\s+order\s+to\b",  # in order to
            r"\bas\s+a\s+result\b",  # as a result
            r"\bon\s+the\s+other\s+hand\b",  # on the other hand
            r"\bfor\s+instance\b",  # for instance
            r"\bin\s+spite\s+of\b",  # in spite of
            r"\bbecause\s+of\b",  # because of
            r"\binstead\s+of\b",  # instead of
            r"\baccording\s+to\b",  # according to
            r"\bas\s+well\s+as\b",  # as well as
        ]

        for pattern in simple_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start, end = match.span()

                # 위치 중복 확인
                if any(abs(start - pos[0]) < 5 for pos in found_positions):
                    continue

                phrase = match.group()
                context = self._get_sentence_context(text, start, end)
                meaning = self.enhanced_korean_definition(
                    phrase, context, is_phrase=True
                )

                results.append(
                    {
                        "original": phrase,
                        "base_form": phrase.lower(),
                        "meaning": meaning,
                        "context": context,
                        "type": "simple_pattern",
                        "is_separated": False,
                        "confidence": 0.7,
                        "user_db_match": False,
                        "match_type": "패턴매칭",
                    }
                )
                found_positions.add((start, end))
                print(f"      ✅ 패턴 매칭: {phrase}")

        return results

    # 🔥 개선된 어려운 단어 추출 (사용자 DB 우선)
    def extract_difficult_words(self, text, easy_words, child_vocab, freq_tiers):
        """사용자 DB 단어 우선 추출 + GPT 필터링"""
        text_str = self._force_extract_text(text)
        word_candidates = []
        user_db_candidates = []  # 🔥 사용자 DB 단어 별도 관리

        # 🔥 원형 기준 중복 추적
        seen_lemmas = set()  # 이미 처리된 원형들
        lemma_to_info = {}  # 원형 → 단어 정보 매핑

        try:
            doc = nlp(text_str)
            for token in doc:
                word = token.text.lower()
                lemma = token.lemma_.lower()
                original_word = token.text

                # 기본 필터링
                if (
                    len(word) < 3
                    or not word.isalpha()
                    or token.is_stop
                    or token.pos_ in ["PUNCT", "SPACE", "SYM"]
                ):
                    continue

                # 🔥 원형 기준 중복 체크
                if lemma in seen_lemmas:
                    continue  # 이미 처리된 원형은 건너뛰기

                seen_lemmas.add(lemma)

                # 문맥 추출
                context = self._get_sentence_context(
                    text_str, token.idx, token.idx + len(token.text)
                )

                word_info = {
                    "word": original_word,  # 원본 형태 (decided)
                    "lemma": lemma,  # 원형 (decide)
                    "context": context,
                    "pos": self._get_simple_pos(token.pos_),
                    "token_info": {
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "original": original_word,
                    },
                }

                # 🔥 사용자 DB 단어 우선 분리 (원형 기준)
                if lemma in self.user_single_words or word in self.user_single_words:
                    user_db_candidates.append(word_info)
                    print(
                        f"      ✅ 사용자 DB 단어 발견: '{original_word}' (원형: {lemma})"
                    )
                else:
                    # 1차 빠른 필터링 (원형 기준)
                    if lemma in easy_words or (child_vocab and lemma in child_vocab):
                        continue
                    word_candidates.append(word_info)

        except Exception as e:
            print(f"❌ 토큰 추출 실패: {e}")
            return []

        final_words = []

        # 🔥 사용자 DB 단어들 처리 (원형으로 의미 생성)
        print(f"   👤 사용자 DB 단어 처리: {len(user_db_candidates)}개")
        for word_info in user_db_candidates:
            korean_meaning = self.enhanced_korean_definition(
                word_info["lemma"],  # 🔥 원형으로 의미 생성
                word_info["context"],
                is_phrase=False,
            )

            word_result = {
                "original": word_info["token_info"]["original"],  # 원본 형태 표시
                "lemma": word_info["lemma"],  # 원형
                "pos": word_info["pos"],
                "korean_meaning": korean_meaning,
                "context": word_info["context"],
                "difficulty_score": 8.0,
                "difficulty_level": "user_priority",
                "confidence": 1.0,
                "inclusion_reason": "사용자DB우선포함",
                "user_db_match": True,
                "is_separated": False,
                "match_type": "사용자DB단어",
                "gpt_filtered": False,
                "english_definition": "",
                "synonyms": "",
                "antonyms": "",
            }
            final_words.append(word_result)

        # 🔥 나머지 단어들도 원형 기준으로 처리
        if word_candidates and hasattr(self, "gpt_filter") and self.gpt_filter:
            print(f"   🤖 GPT 필터링: {len(word_candidates)}개 후보")
            gpt_results = self._process_words_with_gpt_filter(word_candidates)
            final_words.extend(gpt_results)
        else:
            print(f"   ⚙️ 기존 방식: {len(word_candidates)}개 후보")
            traditional_results = self._process_words_without_gpt_filter(
                word_candidates, freq_tiers
            )
            final_words.extend(traditional_results)

        user_db_count = len([w for w in final_words if w.get("user_db_match", False)])
        gpt_count = len([w for w in final_words if not w.get("user_db_match", False)])

        print(
            f"   📊 최종 결과: 사용자DB {user_db_count}개 + 기타 {gpt_count}개 = 총 {len(final_words)}개"
        )

        return final_words

    def is_word_appropriate_for_extraction(self, word, context="", pos=""):
        """통합된 단어 적합성 판별"""

        word_lower = word.lower()

        # 🔥 디버그 출력 추가
        print(f"   🔍 디버그: '{word}' → '{word_lower}'")
        print(f"   🔍 사용자DB크기: {len(self.user_words)}")
        print(f"   🔍 사용자DB포함여부: {word_lower in self.user_words}")
        if len(self.user_words) > 0:
            print(f"   🔍 사용자DB샘플: {list(self.user_words)[:3]}")

        # 사용자 DB에 이미 있는 단어는 무조건 적합
        if word_lower in self.user_words:
            return True, "사용자DB포함"

        # 🔥 1단계: 기본 단어 강력 차단
        if (
            word_lower in BASIC_VERBS
            or word_lower in BASIC_ADJECTIVES
            or word_lower in BASIC_NOUNS
        ):
            return False, "기본단어제외"

        # 🔥 2단계: 외부 DB 기본 어휘 차단
        if is_basic_by_external_db(word_lower):
            return False, "외부DB기본어휘"

        # 🔥 3단계: 사용자 DB 우선 포함
        if word_lower in self.user_single_words:
            return True, "사용자DB우선포함"

        # 🔥 4단계: GPT 문맥별 난이도 분석
        contextual_analysis = self._gpt_analyze_contextual_difficulty(
            word, context, pos
        )
        general_analysis = self._gpt_analyze_word_difficulty(word, context, pos)

        # 두 분석 결과 모두 고려
        return self._determine_final_appropriateness(
            word, contextual_analysis, general_analysis
        )

    def _process_words_with_gpt_filter(self, word_candidates):
        """GPT 필터를 사용한 단어 처리 (사용자 DB 우선 보장)"""
        if not hasattr(self, "gpt_filter") or not self.gpt_filter:
            print("   ❌ GPT 필터가 없어 기존 방식 사용")
            return self._process_words_without_gpt_filter(word_candidates, {})

        print(f"   🤖 GPT 필터링 모드: {len(word_candidates)}개 후보")

        # 🔥 사용자 DB 단어 사전 분리
        user_db_words = []
        regular_words = []

        for word_info in word_candidates:
            word_lower = word_info["word"].lower().strip()
            lemma_lower = word_info["lemma"].lower().strip()

            if (
                word_lower in self.gpt_filter.user_words
                or lemma_lower in self.gpt_filter.user_words
            ):
                user_db_words.append(word_info)
                print(f"      👤 사용자 DB 확인: '{word_info['word']}'")
            else:
                regular_words.append(word_info)

        final_words = []

        # 사용자 DB 단어들 무조건 포함
        for word_info in user_db_words:
            korean_meaning = self.enhanced_korean_definition(
                word_info["lemma"], word_info["context"], is_phrase=False
            )

            word_result = {
                "original": word_info["token_info"]["original"],
                "lemma": word_info["lemma"],
                "pos": word_info["pos"],
                "korean_meaning": korean_meaning,
                "context": word_info["context"],
                "difficulty_score": 8.0,
                "difficulty_level": "user_priority",
                "confidence": 1.0,
                "inclusion_reason": "사용자DB우선포함",
                "user_db_match": True,
                "is_separated": False,
                "match_type": "사용자DB우선",
                "gpt_filtered": False,
                "english_definition": "",
                "synonyms": "",
                "antonyms": "",
            }
            final_words.append(word_result)

        # 🔥 나머지 단어들만 GPT 필터링
        if regular_words:
            words_for_filtering = []
            for word_info in regular_words:
                words_for_filtering.append(
                    {
                        "word": word_info["word"],
                        "context": word_info["context"],
                        "pos": word_info["pos"],
                    }
                )

            try:
                print(f"   📞 GPT 배치 필터링 시작: {len(regular_words)}개...")
                filter_results = self.gpt_filter.batch_filter_words(words_for_filtering)

                gpt_selected_count = 0
                for i, filter_result in enumerate(filter_results):
                    if filter_result["appropriate"]:
                        word_info = regular_words[i]
                        korean_meaning = self.enhanced_korean_definition(
                            word_info["lemma"],  # 🔥 원형으로 의미 생성
                            word_info["context"],
                            is_phrase=False,
                        )

                        word_result = {
                            "original": word_info["token_info"]["original"],
                            "lemma": word_info["lemma"],
                            "pos": word_info["pos"],
                            "korean_meaning": korean_meaning,
                            "context": word_info["context"],
                            "difficulty_score": 0.0,
                            "difficulty_level": "gpt_selected",
                            "confidence": 0.8,
                            "inclusion_reason": filter_result["reason"],
                            "user_db_match": False,
                            "is_separated": False,
                            "match_type": "GPT필터승인",
                            "gpt_filtered": True,
                            "english_definition": "",
                            "synonyms": "",
                            "antonyms": "",
                        }
                        final_words.append(word_result)
                        gpt_selected_count += 1

                print(
                    f"   ✅ GPT 필터링 완료: {gpt_selected_count}/{len(regular_words)}개 선택"
                )

            except Exception as e:
                print(f"   ❌ GPT 필터링 실패: {e}")

        user_db_count = len(user_db_words)
        gpt_count = len([w for w in final_words if not w.get("user_db_match", False)])

        print(
            f"   📊 필터링 결과: 사용자DB {user_db_count}개 + GPT {gpt_count}개 = 총 {len(final_words)}개"
        )

        return final_words

    def _process_words_without_gpt_filter(self, word_candidates, freq_tiers):
        """GPT 필터 없이 기존 방식으로 단어 처리"""

        print(f"   ⚙️ 기존 방식 필터링: {len(word_candidates)}개 후보")

        final_words = []

        for word_info in word_candidates:
            word = word_info["word"]
            lemma = word_info["lemma"]  # 🔥 원형 사용
            context = word_info["context"]
            pos = word_info["pos"]

            # 통합 분석 사용 (원형으로)
            comprehensive_result = self.integrated_comprehensive_analysis(
                lemma, context, pos  # 🔥 원형으로 분석
            )

            # 포함 여부 결정
            should_include, reason = self._should_include_word_comprehensive(
                lemma, comprehensive_result  # 🔥 원형 기준 판단
            )

            if should_include:
                word_result = {
                    "original": word_info["token_info"]["original"],  # 원본 형태
                    "lemma": lemma,  # 원형
                    "pos": pos,
                    "korean_meaning": comprehensive_result["basic_info"][
                        "korean_meaning"
                    ],
                    "context": context,
                    "difficulty_score": comprehensive_result["difficulty"]["score"],
                    "difficulty_level": comprehensive_result["difficulty"]["level"],
                    "recommendation": comprehensive_result["difficulty"][
                        "recommendation"
                    ],
                    "quality_score": comprehensive_result["quality"]["quality_score"],
                    "confidence": comprehensive_result["confidence"]["certainty"],
                    "needs_verification": comprehensive_result["confidence"][
                        "needs_verification"
                    ],
                    "inclusion_reason": reason,
                    "user_db_match": comprehensive_result["user_db_info"][
                        "user_priority"
                    ],
                    "is_separated": comprehensive_result["quality"]["is_separable"],
                    "comprehensive_analysis": True,  # 표시용
                    # 기존 필드들 추가
                    "match_type": "통합분석",
                    "gpt_filtered": False,
                    "difficulty_factors": [],
                    "english_definition": "",
                    "synonyms": ", ".join(
                        comprehensive_result["basic_info"]["synonyms"]
                    ),
                    "antonyms": ", ".join(
                        comprehensive_result["basic_info"]["antonyms"]
                    ),
                }
                final_words.append(word_result)

        user_matches = len([w for w in final_words if w["user_db_match"]])
        print(f"   📊 기존 방식 결과: {len(final_words)}개 선택")
        print(f"   👤 사용자 DB 매칭: {user_matches}개")

        return final_words

    def _legacy_is_difficult(self, word, freq_tiers):
        """기존 방식의 난이도 판별"""
        if len(word) < self.MIN_WORD_LENGTH:
            return False
        if word in freq_tiers:
            return freq_tiers[word] >= self.DIFFICULTY_THRESHOLD
        synsets = wordnet.synsets(word)
        return len(synsets) > 0

    def _force_extract_text(self, text, max_depth=5):
        """텍스트 추출"""
        depth = 0
        current = text
        while isinstance(current, dict) and depth < max_depth:
            if "지문" in current:
                current = current["지문"]
            else:
                current = list(current.values())[0] if current else ""
            depth += 1
        return str(current)

    def _get_sentence_context(self, text, start, end):
        """문장 컨텍스트 추출"""
        text_str = self._force_extract_text(text) if not isinstance(text, str) else text
        left = text_str.rfind(".", 0, start)
        right = text_str.find(".", end)

        if left != -1 and right != -1:
            return text_str[left + 1 : right].strip()
        elif left != -1:
            return text_str[left + 1 :].strip()
        elif right != -1:
            return text_str[:right].strip()
        else:
            return text_str.strip()

    def _get_simple_pos(self, spacy_pos):
        """SpaCy 품사를 한국어로 변환"""
        return {"NOUN": "명사", "VERB": "동사", "ADJ": "형용사", "ADV": "부사"}.get(
            spacy_pos, "기타"
        )

    # 🔥 2. add_synonyms_antonyms_to_results 메서드 완전 교체
    def add_synonyms_antonyms_to_results(self, results):
        """문맥 기반 동의어/반의어 정밀 추출 (최대 3개 제한)"""

        if not hasattr(self, "synonym_extractor") or not self.synonym_extractor:
            print("⚠️ 동의어/반의어 추출기가 없습니다")
            return self._add_empty_synonym_columns(results)

        print("🔍 문맥 기반 동의어/반의어 정밀 추출 중... (최대 3개 제한)")

        # 문맥 기반 추출기인지 확인
        if hasattr(self.synonym_extractor, "enhanced_process_vocabulary"):
            return self._extract_with_contextual_precision(results)
        else:
            return self._extract_with_enhanced_filtering(results)

    def _extract_with_gpt_module(self, results):
        """GPT 모듈을 사용한 추출 (개선된 버전)"""
        try:
            word_list = []
            for result in results:
                word_info = {
                    "word": result.get("단어", ""),
                    "context": result.get("문맥", ""),
                    "pos": result.get("품사", ""),
                    "meaning": result.get("뜻(한글)", ""),
                }
                word_list.append(word_info)

            synonym_results = self.synonym_extractor.batch_extract(word_list)

            for result in results:
                word = result.get("단어", "")
                if word in synonym_results:
                    syn_data = synonym_results[word]

                    # 🔥 추가 필터링 적용
                    raw_synonyms = syn_data.get("synonyms", [])
                    raw_antonyms = syn_data.get("antonyms", [])

                    # 엄격한 필터링 적용
                    filtered_synonyms = enhanced_filter_synonyms_antonyms(
                        raw_synonyms, word, max_count=3
                    )
                    filtered_antonyms = enhanced_filter_synonyms_antonyms(
                        raw_antonyms, word, max_count=2
                    )

                    result["동의어"] = ", ".join(filtered_synonyms)
                    result["반의어"] = ", ".join(filtered_antonyms)
                else:
                    result["동의어"] = ""
                    result["반의어"] = ""

            print("✅ GPT 기반 추출 완료 (필터링 적용)")
            return results

        except Exception as e:
            print(f"❌ GPT 추출 실패: {e}, WordNet으로 전환")
            return self._extract_with_wordnet_enhanced(results)

    def _extract_with_contextual_precision(self, results):
        """문맥 기반 정밀 추출"""
        try:
            print("   🎯 문맥 기반 정밀 모드 사용")

            # vocabulary 형태로 변환
            vocabulary = []
            for result in results:
                vocab_item = {
                    "word": result.get("단어", result.get("original", "")),
                    "context": result.get("문맥", result.get("context", "")),
                    "pos": result.get("품사", result.get("pos", "")),
                    "meaning": result.get("뜻(한글)", result.get("korean_meaning", "")),
                }
                vocabulary.append(vocab_item)

            # 문맥 기반 배치 처리
            enhanced_vocab = self.synonym_extractor.enhanced_process_vocabulary(
                vocabulary
            )

            # 결과에 반영
            for i, result in enumerate(results):
                if i < len(enhanced_vocab):
                    enhanced_item = enhanced_vocab[i]

                    # 🔥 최대 3개 제한 적용
                    synonyms = enhanced_item.get("contextual_synonyms", [])[:3]
                    antonyms = enhanced_item.get("contextual_antonyms", [])[:2]

                    result["동의어"] = ", ".join(synonyms) if synonyms else ""
                    result["반의어"] = ", ".join(antonyms) if antonyms else ""

                    # 추가 정보
                    result["의미정확도"] = enhanced_item.get("meaning_accuracy", 0)
                    result["문맥적합도"] = enhanced_item.get("contextual_fitness", 0)

            contextual_count = len(
                [r for r in results if r.get("동의어") or r.get("반의어")]
            )
            print(f"   ✅ 문맥 기반 정밀 추출 완료: {contextual_count}개 항목")

            return results

        except Exception as e:
            print(f"   ❌ 문맥 기반 추출 실패: {e}, 기존 방식으로 전환")
            return self._extract_with_enhanced_filtering(results)

    def _extract_with_enhanced_filtering(self, results):
        """기존 방식 + 향상된 필터링"""
        try:
            print("   ⚙️ 기존 방식 + 향상된 필터링 모드")

            word_list = []
            for result in results:
                word_info = {
                    "word": result.get("단어", result.get("original", "")),
                    "context": result.get("문맥", result.get("context", "")),
                    "pos": result.get("품사", result.get("pos", "")),
                    "meaning": result.get("뜻(한글)", result.get("korean_meaning", "")),
                }
                word_list.append(word_info)

            # 기존 배치 추출
            synonym_results = self.synonym_extractor.batch_extract(word_list)

            for result in results:
                word = result.get("단어", result.get("original", ""))
                if word in synonym_results:
                    syn_data = synonym_results[word]

                    # 🔥 향상된 필터링 적용
                    raw_synonyms = syn_data.get("synonyms", [])
                    raw_antonyms = syn_data.get("antonyms", [])

                    # 문맥 적합성 검사 (간단 버전)
                    context = result.get("문맥", result.get("context", ""))
                    filtered_synonyms = self._simple_contextual_filter(
                        raw_synonyms, word, context, max_count=3
                    )
                    filtered_antonyms = self._simple_contextual_filter(
                        raw_antonyms, word, context, max_count=2
                    )

                    result["동의어"] = ", ".join(filtered_synonyms)
                    result["반의어"] = ", ".join(filtered_antonyms)
                else:
                    result["동의어"] = ""
                    result["반의어"] = ""

            print("   ✅ 향상된 필터링 완료")
            return results

        except Exception as e:
            print(f"   ❌ 향상된 필터링 실패: {e}")
            return self._add_empty_synonym_columns(results)

    def _simple_contextual_filter(
        self, candidates, original_word, context, max_count=3
    ):
        """간단한 문맥 기반 필터링"""
        if not candidates:
            return []

        # 🔥 기본 필터링 (두 개 이상 단어, 특수문자 제거)
        filtered = []
        for candidate in candidates:
            candidate = candidate.strip()

            # 두 개 이상 단어 제거
            if " " in candidate or "-" in candidate or "_" in candidate:
                continue

            # 숫자나 특수문자 포함 제거
            if not candidate.replace("'", "").isalpha():
                continue

            # 길이 제한
            if len(candidate) < 3 or len(candidate) > 12:
                continue

            # 원본과 너무 유사한 것 제거
            if candidate.lower() == original_word.lower():
                continue

            filtered.append(candidate)

        # 🔥 어근 중복 제거
        unique_filtered = self._remove_root_duplicates(filtered, original_word)

        # 🔥 개수 제한
        return unique_filtered[:max_count]

    def _remove_root_duplicates(self, words, original_word):
        """어근 중복 제거"""
        try:
            from nltk.stem import PorterStemmer

            ps = PorterStemmer()

            seen_stems = set()
            unique_words = []

            # 원본 단어의 어근도 추가
            original_stem = ps.stem(original_word.lower())
            seen_stems.add(original_stem)

            for word in words:
                stem = ps.stem(word.lower())
                if stem not in seen_stems:
                    unique_words.append(word)
                    seen_stems.add(stem)

            return unique_words
        except:
            # NLTK 오류 시 기본 중복 제거만
            return list(dict.fromkeys(words))

    def _add_empty_synonym_columns(self, results):
        """동의어/반의어 빈 컬럼 추가"""
        for result in results:
            result["동의어"] = ""
            result["반의어"] = ""
        return results

    def _extract_with_wordnet_enhanced(self, results):
        """WordNet을 사용한 fallback 추출 (개선된 버전)"""
        print("🔍 동의어/반의어 추가 중... (필터링 적용)")

        try:
            from nltk.corpus import wordnet

            for result in results:
                word = result.get("단어", "").lower()

                # 기본값 설정
                result["동의어"] = ""
                result["반의어"] = ""

                if word and len(word) > 2:
                    try:
                        synonyms = set()
                        antonyms = set()

                        # WordNet에서 동의어/반의어 추출
                        for syn in wordnet.synsets(word):
                            for lemma in syn.lemmas():
                                # 동의어
                                if lemma.name().lower() != word:
                                    synonyms.add(lemma.name().replace("_", " "))

                                # 반의어
                                for antonym in lemma.antonyms():
                                    antonyms.add(antonym.name().replace("_", " "))

                        # 🔥 엄격한 필터링 적용
                        filtered_synonyms = enhanced_filter_synonyms_antonyms(
                            list(synonyms), word, max_count=3
                        )
                        filtered_antonyms = enhanced_filter_synonyms_antonyms(
                            list(antonyms), word, max_count=2
                        )

                        # 결과에 추가
                        if filtered_synonyms:
                            result["동의어"] = ", ".join(filtered_synonyms)
                        if filtered_antonyms:
                            result["반의어"] = ", ".join(filtered_antonyms)

                    except Exception as e:
                        if self.verbose:
                            print(f"   ⚠️ {word} 처리 실패: {e}")
                        continue

        except Exception as e:
            print(f"⚠️ 동의어/반의어 처리 실패: {e}")  # 실패하면 모든 결과에 빈 값 설정
            for result in results:
                result["동의어"] = ""
                result["반의어"] = ""

        return results

    # 🔥 개선된 텍스트 처리 (고급 패턴 분석)
    def process_text(self, text, text_id, easy_words, child_vocab, freq_tiers):
        """개선된 텍스트 처리 - 사용자 DB 단어 우선 보장"""
        text_str = self._force_extract_text(text)
        rows = []

        if self.gpt_token_usage["total_tokens"] >= self.MAX_TOKENS:
            print(f"⚠️ 토큰 사용량 한도 도달. 지문 {text_id} 처리 제한됨.")
            return []

        print(f"📝 지문 {text_id} 처리 시작 (사용자 DB 우선) - 길이: {len(text_str)}자")

        excluded_words = set()

        # 통계 카운터
        stats = {
            "user_db_idioms": 0,
            "grammar_patterns": 0,
            "reference_idioms": 0,
            "user_db_words": 0,  # 🔥 사용자 DB 단어
            "gpt_words": 0,  # 🔥 GPT 선택 단어
            "advanced_phrasal_verbs": 0,
            "simple_patterns": 0,
        }

        # 1. 숙어 추출
        print(f"🔍 지문 {text_id}에서 숙어 추출 중...")
        idioms = self.extract_advanced_idioms(text)

        for idiom in idioms:
            for word in idiom["base_form"].lower().split():
                if word not in ["~", "n", "v", "v-ing"]:
                    excluded_words.add(word)

            # 통계 업데이트
            if idiom["type"] == "user_db_idiom":
                stats["user_db_idioms"] += 1
            elif idiom["type"] == "grammar_pattern":
                stats["grammar_patterns"] += 1
            elif idiom["type"] == "reference_idiom_db":
                stats["reference_idioms"] += 1
            elif "advanced_phrasal" in idiom["type"]:
                stats["advanced_phrasal_verbs"] += 1
            elif idiom["type"] == "simple_pattern":
                stats["simple_patterns"] += 1

            rows.append(
                {
                    "지문ID": text_id,
                    "지문": text_str,
                    "단어": idiom["original"],
                    "원형": idiom["base_form"],
                    "품사": idiom["type"],
                    "뜻(한글)": idiom["meaning"],
                    "뜻(영어)": "",
                    "동의어": "",
                    "반의어": "",
                    "문맥": idiom["context"],
                    "분리형여부": idiom["is_separated"],
                    "신뢰도": f"{idiom['confidence']:.2f}",
                    "사용자DB매칭": idiom.get("user_db_match", False),
                    "매칭방식": idiom.get("match_type", ""),
                    "패턴정보": idiom.get("pattern_info", {}),
                }
            )

        # 2. 어려운 단어 추출 (사용자 DB 우선)
        if self.gpt_token_usage["total_tokens"] < self.MAX_TOKENS:
            print(f"🔍 지문 {text_id}에서 어려운 단어 추출 중... (사용자 DB 최우선)")
            try:
                difficult_words = self.extract_difficult_words(
                    text, easy_words, child_vocab, freq_tiers
                )

                for word in difficult_words:
                    if word.get("lemma", "").lower() in excluded_words:
                        continue

                    # 🔥 통계 분류 개선
                    if word.get("user_db_match", False):
                        stats["user_db_words"] += 1
                    else:
                        stats["gpt_words"] += 1

                    rows.append(
                        {
                            "지문ID": text_id,
                            "지문": text_str,
                            "단어": word.get("original", ""),
                            "원형": word.get("lemma", ""),
                            "품사": word.get("pos", ""),
                            "뜻(한글)": word.get("korean_meaning", ""),
                            "뜻(영어)": word.get("english_definition", ""),
                            "동의어": word.get("synonyms", ""),
                            "반의어": word.get("antonyms", ""),
                            "문맥": word.get("context", ""),
                            "분리형여부": word.get("is_separated", False),
                            "신뢰도": f"{word.get('confidence', 0):.2f}",
                            "사용자DB매칭": word.get("user_db_match", False),
                            "매칭방식": word.get("match_type", ""),
                            "포함이유": word.get("inclusion_reason", ""),
                            "난이도점수": f"{word.get('difficulty_score', 0):.2f}",
                            "난이도레벨": word.get("difficulty_level", "medium"),
                        }
                    )
            except Exception as e:
                if self.verbose:
                    print(f"❌ 어려운 단어 추출 실패: {e}")

        # 🔥 개선된 통계 출력
        print(f"✅ 지문 {text_id} 처리 완료:")
        print(f"   📊 사용자 DB 숙어: {stats['user_db_idioms']}개")
        print(f"   📊 문법 패턴: {stats['grammar_patterns']}개")
        print(f"   📊 참조 DB 숙어: {stats['reference_idioms']}개")
        print(f"   📊 사용자 DB 단어: {stats['user_db_words']}개 ✅")  # 🔥 강조
        print(f"   📊 GPT 선택 단어: {stats['gpt_words']}개")
        print(f"   📊 고급 구동사: {stats['advanced_phrasal_verbs']}개")

        # 동의어/반의어 추가
        if rows:
            try:
                rows = self.add_synonyms_antonyms_to_results(rows)
                print(f"✅ 동의어/반의어 추가 완료")
            except Exception as e:
                print(f"⚠️ 동의어/반의어 추가 실패: {e}")

        return rows

    # 🔥 고급 패턴 분석 단어장 생성
    def generate_vocabulary_workbook(
        self, texts, output_file="vocabulary_advanced.xlsx", **kwargs
    ):
        """고급 패턴 분석 단어장 생성"""
        start_time = time.time()

        print(f"🚀 고급 패턴 분석 단어장 생성 시작")
        print(f"   • 사용자 DB 숙어: ✅ {len(self.user_idioms)}개 활용")
        print(f"   • 사용자 DB 단어: ✅ {len(self.user_single_words)}개 활용")
        print(f"   • 문법 패턴 분석: ✅ V-ing, N V-ing 등")  # 🔥 새로 추가
        print(f"   • 고급 구동사 분석: ✅ 연속형/분리형 자동 구분")  # 🔥 새로 추가
        print(f"   • 참조 숙어 DB: ✅ 활용")

        # 진행률 표시와 함께 텍스트 처리
        results = []
        for idx, text in enumerate(
            tqdm(texts, desc="📝 텍스트 처리 중 (고급 패턴 분석)", unit="지문")
        ):
            try:
                result = self.process_text(
                    text, idx + 1, self.easy_words, set(), self.freq_tiers
                )
                results.extend(result)

                if (idx + 1) % 10 == 0:
                    tqdm.write(
                        f"✅ {idx + 1}/{len(texts)} 완료 ({len(results)}개 항목 추출)"
                    )
                    tqdm.write(
                        f"   📊 GPT 호출: {self.gpt_call_count}/{self.GPT_CALL_LIMIT}"
                    )
                    tqdm.write(
                        f"   📊 토큰 사용: {self.gpt_token_usage['total_tokens']}/{self.MAX_TOKENS}"
                    )

            except Exception as e:
                tqdm.write(f"❌ 텍스트 {idx + 1} 처리 실패: {e}")
                continue

        if not results:
            print("⚠️ 추출된 단어/숙어가 없습니다.")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # 품질 검사
        quality_results = None
        if kwargs.get("enable_quality_check", True):
            df, quality_results = self.run_quality_check_and_fix(df, output_file)

        # 사용자 DB 매칭 분석
        if len(df) > 0:
            self._analyze_user_db_matching(df)

        # Excel 저장
        try:
            df.to_excel(output_file, index=False)
            print(f"✅ 단어장 저장 완료: {output_file}")
        except Exception as e:
            print(f"❌ Excel 저장 실패: {e}")

        # 캐시 저장
        self.save_cache_to_file()

        # 결과 요약
        processing_time = time.time() - start_time
        print(f"\n🎯 고급 패턴 분석 단어장 생성 결과:")
        print(f"   ⏱️ 처리 시간: {processing_time:.2f}초")
        print(f"   📊 총 항목 수: {len(df)}개")
        print(f"   📊 GPT 호출 횟수: {self.gpt_call_count}회")
        print(f"   📊 토큰 사용량: {self.gpt_token_usage['total_tokens']}개")

        if self.settings["USE_INTEGRATED_CONTEXTUAL"]:
            print(f"   ✨ 통합 컨텍스트 의미 생성으로 품질 향상")
        if self.settings["USE_INTEGRATED_DIFFICULTY"]:
            print(f"   ✨ 통합 난이도 분석으로 정확도 향상")

        print(f"   🔥 사용자 DB 우선 + 고급 패턴 분석으로 정밀한 단어장 생성")

        # 🔥 패턴별 통계 출력
        if "매칭방식" in df.columns:
            pattern_stats = df["매칭방식"].value_counts()
            print(f"\n📊 패턴별 추출 통계:")
            for pattern, count in pattern_stats.items():
                if pattern:
                    print(f"   • {pattern}: {count}개")

        return df

    def _analyze_user_db_matching(self, df):
        """사용자 DB 매칭 분석"""

        try:
            if "사용자DB매칭" in df.columns:
                user_matched = df["사용자DB매칭"].sum()
                total_items = len(df)
                match_ratio = (
                    (user_matched / total_items * 100) if total_items > 0 else 0
                )

                print(f"   👤 사용자 DB 매칭 항목: {user_matched}개")
                print(f"   📊 전체 대비 비율: {match_ratio:.1f}%")

                # 매칭 방식별 분포
                if "매칭방식" in df.columns and user_matched > 0:
                    user_df = df[df["사용자DB매칭"] == True]
                    match_types = user_df["매칭방식"].value_counts()
                    print(f"   🔍 매칭 방식별 분포:")
                    for match_type, count in match_types.items():
                        if match_type:
                            print(f"      • {match_type}: {count}개")

                # 포함 이유별 분포
                if "포함이유" in df.columns:
                    inclusion_reasons = df["포함이유"].value_counts()
                    print(f"   📋 포함 이유별 분포:")
                    for reason, count in inclusion_reasons.items():
                        if reason:
                            print(f"      • {reason}: {count}개")

        except Exception as e:
            print(f"   ❌ 사용자 DB 매칭 분석 실패: {e}")

    def run_quality_check_and_fix(self, df, output_file):
        """품질 검사 및 자동 수정"""
        print("\n🔍 품질 검사 시작...")

        try:
            temp_file = output_file.replace(".xlsx", "_temp.xlsx")
            df.to_excel(temp_file, index=False)

            checker = VocabularyQualityChecker(temp_file)
            results = checker.generate_quality_report()

            print("🔧 품질 문제 자동 수정 중...")
            fixed_df = checker.update_vocabulary_with_fixes()

            if os.path.exists(temp_file):
                os.remove(temp_file)

            print(f"📊 품질 점수: {results['quality_score']:.1f}/100")
            print(f"📊 발견된 문제: {results['total_issues']}개")

            return fixed_df, results

        except Exception as e:
            print(f"❌ 품질 검사 실패: {e}")
            return df, None

    def _gpt_analyze_contextual_difficulty(self, word, context, pos=""):
        """지문 문맥에서의 특정 의미 난이도 분석"""

        prompt = f"""
Analyze the difficulty of the word "{word}" specifically as used in this context for Korean high school students.

Context: "{context}"

Focus on:
1. What does "{word}" mean in THIS specific context?
2. Is THIS particular meaning/usage advanced for high school students?
3. Would students struggle with THIS specific usage (not the word in general)?

Examples of contextual analysis:
- "feel tired" → basic usage (exclude)
- "feel the economic impact" → more advanced usage (consider including)
- "work hard" → basic usage (exclude)  
- "the work explores themes" → academic usage (include)

Rate the CONTEXTUAL difficulty (1-10):
1-4: Basic usage that high school students know
5-6: Intermediate usage (borderline)
7-10: Advanced/academic usage that students might not know

JSON response:
{{
    "contextual_meaning": "specific meaning in this context",
    "contextual_difficulty": 1-10,
    "usage_type": "basic|intermediate|advanced|academic",
    "recommendation": "include|exclude",
    "reasoning": "why this specific usage is/isn't challenging"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in contextual English analysis for Korean students.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()

            return json.loads(content)
        except Exception as e:
            return {
                "contextual_difficulty": 5,
                "recommendation": "exclude",
                "reasoning": f"분석 실패: {e}",
            }

    def _determine_final_appropriateness(
        self, word, contextual_analysis, general_analysis
    ):
        """최종 적합성 판별"""

        if not contextual_analysis or not general_analysis:
            return False, "분석실패"

        # 문맥 분석 우선
        contextual_difficulty = contextual_analysis.get("contextual_difficulty", 5)
        general_difficulty = general_analysis.get("difficulty_score", 5)

        # 엄격한 기준 적용
        if contextual_difficulty >= 6 and general_difficulty >= 7:
            return (
                True,
                f"고난이도확인(문맥{contextual_difficulty},일반{general_difficulty})",
            )
        elif contextual_difficulty >= 6 and general_difficulty >= 8:
            return (
                True,
                f"일반고난이도(문맥{contextual_difficulty},일반{general_difficulty})",
            )
        else:
            return (
                False,
                f"난이도부족(문맥{contextual_difficulty},일반{general_difficulty})",
            )


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="고급 패턴 분석 단어장 생성기 v4.0")
    parser.add_argument("--input", "-i", default="지문DB.csv", help="입력 CSV 파일")
    parser.add_argument(
        "--output", "-o", default="vocabulary_advanced.xlsx", help="출력 엑셀 파일"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=2.5, help="난이도 임계값"
    )
    parser.add_argument("--cache", "-c", default="gpt_cache.json", help="GPT 캐시 파일")
    parser.add_argument(
        "--max-tokens", "-mt", type=int, default=200000, help="최대 토큰 사용량"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="상세한 로그 출력")
    parser.add_argument("--user-words", help="사용자 단어 파일 (CSV/XLSX)")
    parser.add_argument("--data-dir", default="data", help="숙어 데이터 디렉토리")
    parser.add_argument(
        "--no-quality-check", action="store_true", help="품질 검사 건너뛰기"
    )

    args = parser.parse_args()

    settings = {
        "DIFFICULTY_THRESHOLD": args.threshold,
        "CACHE_FILE": args.cache,
        "MAX_TOKENS": args.max_tokens,
        "USE_CACHE": True,
        "USE_INTEGRATED_CONTEXTUAL": True,
        "USE_INTEGRATED_DIFFICULTY": True,
        "data_dir": args.data_dir,
    }

    print(f"🚀 고급 패턴 분석 단어장 생성기 v4.0")
    print(f"   • 사용자 DB 숙어 우선 인식: ✅")
    print(f"   • 사용자 DB 단어 우선 포함: ✅")
    print(f"   • 문법 패턴 분석: ✅ (V-ing)")  # 🔥 새로 추가
    print(f"   • 고급 구동사 분석: ✅ (연속형/분리형 자동 구분)")  # 🔥 새로 추가
    print(f"   • 분리형 표시 개선: ✅ (pick ~ up, spend time V-ing)")  # 🔥 새로 추가
    print(f"   • 통합 컨텍스트 의미: ✅")
    print(f"   • 통합 난이도 분석: ✅")

    try:
        extractor = AdvancedVocabExtractor(
            user_words_file=args.user_words if args.user_words else "단어DB.csv",
            settings=settings,
            csv_file=args.input,
            verbose=args.verbose,
        )

        # 입력 파일 로드
        if args.input.endswith(".xlsx"):
            text_df = pd.read_excel(args.input)
        else:
            try:
                text_df = pd.read_csv(args.input, encoding="utf-8")
            except UnicodeDecodeError:
                text_df = pd.read_csv(args.input, encoding="cp949")

        text_column = "content" if "content" in text_df.columns else text_df.columns[0]
        print(f"✅ '{args.input}' 파일에서 '{text_column}' 열을 사용")

        texts = text_df[text_column].dropna().astype(str).tolist()

        print(f"📚 총 {len(texts)}개 텍스트에서 고급 패턴 분석 단어장 생성 시작")

        df = extractor.generate_vocabulary_workbook(
            texts,
            output_file=args.output,
            enable_quality_check=not args.no_quality_check,
        )

        # 결과 출력
        if df is not None and len(df) > 0:
            print(f"\n🎉 고급 패턴 분석 단어장 생성 완료!")
            print(f"   📁 파일: {args.output}")
            print(f"   📊 총 항목: {len(df)}개")
            print(
                f"   ✨ 고급 패턴 분석 적용 (분리형: pick ~ up, 문법패턴: spend time V-ing)"
            )

            # 사용자 DB 매칭 통계 출력
            if "사용자DB매칭" in df.columns:
                user_matched = df["사용자DB매칭"].sum()
                total_items = len(df)
                match_ratio = (
                    (user_matched / total_items * 100) if total_items > 0 else 0
                )
                print(f"\n👤 사용자 DB 매칭 결과:")
                print(f"   • 매칭된 항목: {user_matched}개")
                print(f"   • 전체 대비 비율: {match_ratio:.1f}%")

            # 🔥 패턴별 통계 출력
            if "매칭방식" in df.columns:
                pattern_stats = df["매칭방식"].value_counts()
                print(f"\n📊 패턴별 추출 통계:")
                for pattern, count in pattern_stats.items():
                    if pattern:
                        print(f"   • {pattern}: {count}개")

        else:
            print("⚠️ 단어장 생성에 실패했거나 추출된 항목이 없습니다.")

    except FileNotFoundError:
        print(f"❌ '{args.input}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "help":
        print(
            """
    고급 패턴 분석 단어장 생성기 v4.0 사용법:

    기본 사용법:
    python improved_vocab_extractor.py --input 지문DB.csv --output vocabulary.xlsx

    🔥 주요 개선사항 (v4.0):
    ✅ 고급 구동사 패턴 분석
    • 연속형 가능: pick up (그대로 표시)
    • 분리 필수: pick ~ up (~ 표시)
    • 실제 분리: pick something up → pick ~ up

    ✅ 문법 패턴 자동 인식
    • V-ing 패턴: spend time reading → spend time V-ing
    • N V-ing 패턴: prevent him from going → prevent N from V-ing

    ✅ 사용자 DB 우선 처리
    • 사용자 DB 숙어 최우선 인식
    • 사용자 DB 단어 우선 포함
    • 매칭 방식별 상세 통계

    ✅ 패턴별 정밀 분석
    • 위치 기반 중복 방지
    • 문법적 검증 강화
    • 신뢰도 기반 우선순위

    패턴 표시 예시:
    - pick up → pick up (연속형 가능)
    - pick something up → pick ~ up (분리 필수)
    - spend time reading → spend time V-ing (문법 패턴)

    주요 옵션:
    --input: 입력 CSV 파일 (기본: 지문DB.csv)
    --output: 출력 Excel 파일 (기본: vocabulary_advanced.xlsx)
    --user-words: 사용자 단어 파일 (기본: 단어DB.csv)
    --data-dir: 참조 숙어 데이터 디렉토리 (기본: data)
    --verbose: 상세한 로그 출력
    --no-quality-check: 품질 검사 건너뛰기

    사용 예시:
    python improved_vocab_extractor.py --input 지문DB.csv --output my_vocab.xlsx --verbose
        """
        )
        sys.exit(0)

    try:
        print("🚀 고급 패턴 분석 단어장 생성기 시작...")
        print("🔥 v4.0 - 분리형/문법패턴 고급 분석 버전")
        main()
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
