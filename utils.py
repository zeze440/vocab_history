# utils.py - 유틸리티 함수들

import re
import os
import time
import json
import pandas as pd
from config import ENCODING_ORDER, TEXT_COLUMNS, POS_MAPPING


def force_extract_text(text, max_depth=5):
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


def get_sentence_context(text, start, end):
    """문장 컨텍스트 추출"""
    text_str = force_extract_text(text) if not isinstance(text, str) else text
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


def get_simple_pos(spacy_pos):
    """SpaCy 품사를 한국어로 변환"""
    return POS_MAPPING.get(spacy_pos, "기타")


def clean_korean_definition(word, text):
    """한글 뜻 정리 함수"""
    if not isinstance(text, str):
        return ""

    text = text.strip("\"'").strip()
    text = (
        text.replace('"', "")
        .replace("'", "")
        .replace(""", "").replace(""", "")
        .replace("'", "")
        .replace("'", "")
    )

    # 숙어/패턴인지 확인
    if " A " in word or " B " in word or " " in word.strip():
        return text  # 영어 표현 유지

    # 일반 단어인 경우 영어 제거
    text = re.sub(r"[a-zA-Z]{3,}", "", text)  # 3글자 이상 영어 단어 제거
    return text.strip()


def safe_read_csv(file_path, encodings=None):
    """안전한 CSV 읽기 (여러 인코딩 시도)"""
    if encodings is None:
        encodings = ENCODING_ORDER

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"   ✅ 인코딩 성공: {encoding}")
            return df
        except UnicodeDecodeError:
            continue

    print(f"   ❌ 모든 인코딩 시도 실패")
    return None


def find_text_column(df):
    """데이터프레임에서 텍스트 컬럼 찾기 (data_loader.py와 통합된 버전 사용)"""
    from data_loader import find_text_column as dl_find_text_column

    return dl_find_text_column(df)


def enhanced_filter_synonyms_antonyms(candidates, original_word, max_count=3):
    """동의어/반의어 향상된 필터링 (어근 중복 제거 + 개수 제한)"""
    if not candidates:
        return []

    # 1단계: 기본 필터링
    basic_filtered = []
    for candidate in candidates:
        candidate = str(candidate).strip()

        # 두 개 이상 단어 강력 차단
        if " " in candidate or "-" in candidate or "_" in candidate:
            continue

        # 숫자나 특수문자 포함 차단 (아포스트로피 제외)
        if not re.match(r"^[a-zA-Z']+$", candidate):
            continue

        # 길이 제한
        if len(candidate) < 3 or len(candidate) > 12:
            continue

        # 원본과 동일한 것 제거
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


def safe_gpt_call(
    client,
    prompt_messages,
    model="gpt-4o",
    max_tokens=300,
    temperature=0.1,
    max_retries=3,
):
    """안전한 GPT API 호출 with 재시도 로직"""
    import openai

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


def load_json_safe(file_path, default=None):
    """안전한 JSON 로드"""
    if default is None:
        default = {}

    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ JSON 로드 실패 ({file_path}): {e}")

    return default


def save_json_safe(data, file_path):
    """안전한 JSON 저장"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ JSON 저장 실패 ({file_path}): {e}")
        return False


def enhanced_filter_synonyms_antonyms(
    synonyms, antonyms, original_word, max_synonyms=3, max_antonyms=2
):
    """동의어/반의어 향상된 필터링"""
    try:
        # 기본 검증
        if not isinstance(synonyms, (list, tuple)):
            synonyms = []
        if not isinstance(antonyms, (list, tuple)):
            antonyms = []

        # 동의어 필터링
        filtered_synonyms = []
        for syn in synonyms:
            if (
                isinstance(syn, str)
                and syn.strip()
                and syn.lower() != original_word.lower()
                and len(syn.strip()) > 2
                and " " not in syn.strip()
            ):  # 단일 단어만
                filtered_synonyms.append(syn.strip())

                if len(filtered_synonyms) >= max_synonyms:
                    break

        # 반의어 필터링
        filtered_antonyms = []
        for ant in antonyms:
            if (
                isinstance(ant, str)
                and ant.strip()
                and ant.lower() != original_word.lower()
                and len(ant.strip()) > 2
                and " " not in ant.strip()
            ):  # 단일 단어만
                filtered_antonyms.append(ant.strip())

                if len(filtered_antonyms) >= max_antonyms:
                    break

        return filtered_synonyms, filtered_antonyms

    except Exception as e:
        print(f"⚠️ 동의어/반의어 필터링 실패: {e}")
        return [], []


def safe_get_dict_value(dictionary, key, default=""):
    """딕셔너리에서 안전하게 값 가져오기"""
    try:
        if isinstance(dictionary, dict):
            return dictionary.get(key, default)
        return default
    except:
        return default


def validate_extraction_result(result):
    """추출 결과 유효성 검사"""
    if not result:
        return False

    if not isinstance(result, dict):
        return False

    # 필수 키 확인
    required_keys = ["original", "meaning"]
    for key in required_keys:
        if key not in result:
            return False

    return True


def clean_text_for_processing(text):
    """텍스트 전처리"""
    try:
        if not text:
            return ""

        text_str = str(text).strip()

        # 과도한 공백 제거
        import re

        text_str = re.sub(r"\s+", " ", text_str)

        # 특수문자 정리 (기본적인 것만)
        text_str = re.sub(r'[^\w\s\.,!?;:\'"()-]', " ", text_str)

        return text_str.strip()

    except Exception as e:
        print(f"⚠️ 텍스트 전처리 실패: {e}")
        return str(text) if text else ""


def safe_list_extend(target_list, source_list):
    """리스트 안전하게 확장"""
    try:
        if not isinstance(target_list, list):
            target_list = []

        if source_list and isinstance(source_list, (list, tuple)):
            target_list.extend(source_list)

        return target_list

    except Exception as e:
        print(f"⚠️ 리스트 확장 실패: {e}")
        return target_list if isinstance(target_list, list) else []
