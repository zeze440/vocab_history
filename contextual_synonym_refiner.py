#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 하이브리드 동의어/반의어 정제 시스템
- VOCABULARY.XLSX와 동일한 구조로 출력
- 무료 사전 API들로 교차 검증하여 후보 수집
- AI 기반 원형 단어 판별
- GPT로 문맥 적합성 및 수준 검증
- 최대 3개까지 엄선
"""

import pandas as pd
import numpy as np
import re
import argparse
import json
import os
import time
import requests
import openai
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from datetime import datetime
import warnings
from dotenv import load_dotenv
import nltk
from nltk.corpus import wordnet

warnings.filterwarnings("ignore")

# .env 파일 로드
load_dotenv()


def check_dependencies():
    """필수 라이브러리 확인"""
    missing = []

    try:
        import nltk
    except ImportError:
        missing.append("nltk")

    try:
        import openai
    except ImportError:
        missing.append("openai")

    if missing:
        print(f"❌ 필수 라이브러리가 설치되지 않음: {', '.join(missing)}")
        print("설치 명령어: pip install " + " ".join(missing))
        return False
    return True


class ContextualSynonymExtractor:
    """문맥을 고려한 정밀 동의어/반의어 추출기"""

    def __init__(self, client, verbose=False):
        self.client = client
        self.verbose = verbose

        # 🔥 이 부분에서 nltk를 import 없이 사용
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("wordnet", quiet=True)

    def extract_contextual_meaning_with_gpt(
        self, word: str, context: str, pos: str = ""
    ) -> Dict:
        """GPT로 문맥에서의 구체적 의미 추출"""

        prompt = f"""
Analyze the word "{word}" in this specific context and determine its EXACT meaning.

Context: "{context}"
Part of Speech: {pos}

Task: Identify the specific meaning/sense of "{word}" as used in THIS context.

Instructions:
1. Look at HOW the word is used in this sentence
2. Determine the specific meaning (not all possible meanings)
3. Consider the surrounding words and overall meaning
4. Focus on the contextual definition

Respond in JSON format:
{{
    "contextual_meaning": "specific meaning of the word in this context",
    "semantic_field": "the domain/field this meaning belongs to (e.g., 'finance', 'nature', 'emotion', 'physical')",
    "usage_type": "how it's used (e.g., 'literal', 'metaphorical', 'technical', 'casual')",
    "meaning_certainty": 0.0-1.0,
    "key_context_clues": ["words or phrases that help determine this meaning"]
}}

Example:
- "bank" in "I went to the bank" → financial institution
- "bank" in "river bank" → edge of water
- "light" in "light weight" → not heavy  
- "light" in "turn on the light" → illumination
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert linguist who identifies specific word meanings in context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()

            # JSON 파싱
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()

            import json

            result = json.loads(content)
            return result

        except Exception as e:
            if self.verbose:
                print(f"⚠️ GPT 의미 추출 실패: {e}")
            return {
                "contextual_meaning": "",
                "semantic_field": "general",
                "usage_type": "literal",
                "meaning_certainty": 0.5,
                "key_context_clues": [],
            }

    def get_wordnet_synsets_by_context(
        self, word: str, contextual_meaning: str, pos: str = ""
    ) -> List:
        """문맥적 의미에 맞는 WordNet synset들만 선별"""

        # 품사 매핑
        pos_mapping = {
            "n": wordnet.NOUN,
            "v": wordnet.VERB,
            "a": wordnet.ADJ,
            "r": wordnet.ADV,
        }
        wordnet_pos = pos_mapping.get(pos.lower(), None)

        # 모든 synset 가져오기
        if wordnet_pos:
            synsets = wordnet.synsets(word, pos=wordnet_pos)
        else:
            synsets = wordnet.synsets(word)

        if not synsets or not contextual_meaning:
            return synsets[:2]  # 기본값: 처음 2개만

        # 문맥적 의미와 synset 정의 비교
        relevant_synsets = []
        contextual_keywords = set(contextual_meaning.lower().split())

        for synset in synsets:
            definition = synset.definition().lower()
            examples = [ex.lower() for ex in synset.examples()]

            # 정의에서 키워드 매칭 점수 계산
            definition_words = set(definition.split())
            example_words = set()
            for ex in examples:
                example_words.update(ex.split())

            # 유사도 계산
            definition_overlap = len(contextual_keywords & definition_words) / max(
                len(contextual_keywords), 1
            )
            example_overlap = (
                len(contextual_keywords & example_words)
                / max(len(contextual_keywords), 1)
                if example_words
                else 0
            )

            total_score = definition_overlap + (example_overlap * 0.5)

            relevant_synsets.append((synset, total_score))

            if self.verbose:
                print(f"   📝 {synset.name()}: {definition} (점수: {total_score:.3f})")

        # 점수순 정렬하고 상위 2개만 선택
        relevant_synsets.sort(key=lambda x: x[1], reverse=True)
        selected_synsets = [
            synset for synset, score in relevant_synsets[:2] if score > 0.1
        ]

        if self.verbose:
            print(f"   ✅ 선택된 synset: {len(selected_synsets)}개")

        return selected_synsets if selected_synsets else synsets[:1]

    def extract_synonyms_from_synsets(
        self, synsets: List, original_word: str
    ) -> List[str]:
        """선별된 synset들에서만 동의어 추출"""

        synonyms = set()
        original_lower = original_word.lower()

        for synset in synsets:
            for lemma in synset.lemmas():
                lemma_name = lemma.name().replace("_", " ")

                # 기본 필터링
                if (
                    lemma_name.lower() != original_lower
                    and " " not in lemma_name  # 단일 단어만
                    and "-" not in lemma_name  # 하이픈 없음
                    and len(lemma_name) >= 3  # 최소 3글자
                    and len(lemma_name) <= 12  # 최대 12글자
                    and lemma_name.isalpha()  # 알파벳만
                ):
                    synonyms.add(lemma_name)

        return list(synonyms)

    def extract_antonyms_from_synsets(
        self, synsets: List, original_word: str
    ) -> List[str]:
        """선별된 synset들에서만 반의어 추출"""

        antonyms = set()
        original_lower = original_word.lower()

        for synset in synsets:
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    antonym_name = antonym.name().replace("_", " ")

                    # 기본 필터링
                    if (
                        antonym_name.lower() != original_lower
                        and " " not in antonym_name  # 단일 단어만
                        and "-" not in antonym_name  # 하이픈 없음
                        and len(antonym_name) >= 3  # 최소 3글자
                        and len(antonym_name) <= 12  # 최대 12글자
                        and antonym_name.isalpha()  # 알파벳만
                    ):
                        antonyms.add(antonym_name)

        return list(antonyms)

    def validate_synonyms_with_context(
        self, word: str, synonyms: List[str], context: str, contextual_meaning: str
    ) -> List[str]:
        """문맥과 의미를 고려한 동의어 검증"""

        if not synonyms or len(synonyms) <= 2:
            return synonyms

        prompt = f"""
Validate these synonym candidates for the word "{word}" in this specific context.

Original word: "{word}"
Context: "{context}"
Specific meaning in context: "{contextual_meaning}"
Synonym candidates: {', '.join(synonyms)}

Task: Select ONLY the synonyms that:
1. Have the SAME specific meaning as "{word}" in this context
2. Can replace "{word}" in this sentence without changing the meaning
3. Are appropriate for Korean high school students
4. Are single words (no phrases)

Test each candidate by substitution:
- Original: "{context}"
- With synonym: Would the meaning stay exactly the same?

Respond in JSON format:
{{
    "validated_synonyms": ["synonym1", "synonym2"],
    "rejected_synonyms": [
        {{"word": "rejected_word", "reason": "why it doesn't fit this context"}},
    ],
    "context_analysis": "brief explanation of the word's meaning in this context"
}}

Be VERY strict - when in doubt, reject the synonym.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict English teacher who only approves synonyms that perfectly match the contextual meaning.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()

            # JSON 파싱
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()

            import json

            result = json.loads(content)
            validated = result.get("validated_synonyms", [])

            if self.verbose:
                rejected = result.get("rejected_synonyms", [])
                print(f"   ✅ 검증 통과: {validated}")
                if rejected:
                    print(f"   ❌ 검증 실패: {[r['word'] for r in rejected]}")

            return validated[:3]  # 최대 3개

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 동의어 검증 실패: {e}")
            return synonyms[:2]  # 기본값으로 처음 2개만

    def extract_contextual_synonyms_antonyms(
        self, word: str, context: str, pos: str = ""
    ) -> Dict:
        """문맥 기반 정밀 동의어/반의어 추출"""
        # 먼저 기본 초기화
        self.client = openai.OpenAI()
        # 나중에 contextual extractor 초기화
        self.contextual_extractor = ContextualSynonymExtractor(
            client=self.client, verbose=self.verbose
        )
        # 1단계: GPT로 문맥적 의미 추출
        meaning_analysis = self.extract_contextual_meaning_with_gpt(word, context, pos)
        contextual_meaning = meaning_analysis.get("contextual_meaning", "")

        if self.verbose:
            print(f"   📝 문맥적 의미: {contextual_meaning}")

        # 2단계: 문맥에 맞는 WordNet synset 선별
        relevant_synsets = self.get_wordnet_synsets_by_context(
            word, contextual_meaning, pos
        )

        # 3단계: 선별된 synset에서만 동의어/반의어 추출
        raw_synonyms = self.extract_synonyms_from_synsets(relevant_synsets, word)
        raw_antonyms = self.extract_antonyms_from_synsets(relevant_synsets, word)

        if self.verbose:
            print(f"   🔍 후보 동의어: {raw_synonyms}")
            print(f"   🔍 후보 반의어: {raw_antonyms}")

        # 4단계: 문맥 기반 검증 (동의어만, 반의어는 이미 정밀함)
        if raw_synonyms:
            validated_synonyms = self.validate_synonyms_with_context(
                word, raw_synonyms, context, contextual_meaning
            )
        else:
            validated_synonyms = []

        # 5단계: 어근 중복 제거
        final_synonyms = self.remove_root_duplicates(validated_synonyms, word)
        final_antonyms = self.remove_root_duplicates(
            raw_antonyms[:2], word
        )  # 반의어는 최대 2개

        return {
            "synonyms": final_synonyms,
            "antonyms": final_antonyms,
            "contextual_meaning": contextual_meaning,
            "semantic_field": meaning_analysis.get("semantic_field", ""),
            "meaning_certainty": meaning_analysis.get("meaning_certainty", 0.5),
            "method": "contextual_precise",
        }

    def remove_root_duplicates(self, words: List[str], original_word: str) -> List[str]:
        """어근 중복 제거 (간단한 버전)"""
        if not words:
            return words

        # 불규칙 어근 매핑
        irregular_roots = {
            "better": "good",
            "best": "good",
            "worse": "bad",
            "worst": "bad",
            "more": "much",
            "most": "much",
            "less": "little",
            "least": "little",
            "further": "far",
            "furthest": "far",
            "older": "old",
            "oldest": "old",
        }

        def get_root(word):
            word_lower = word.lower()
            if word_lower in irregular_roots:
                return irregular_roots[word_lower]

            # 간단한 접미사 제거
            suffixes = ["ing", "ed", "er", "est", "ly", "ness", "ment", "tion"]
            for suffix in suffixes:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                    return word_lower[: -len(suffix)]
            return word_lower

        filtered = []
        seen_roots = set()
        original_root = get_root(original_word)

        for word in words:
            word_root = get_root(word)

            # 원본과 같은 어근이거나 이미 본 어근이면 제외
            if word_root == original_root or word_root in seen_roots:
                continue

            seen_roots.add(word_root)
            filtered.append(word)

        return filtered

    def integrate_contextual_extractor(self):
        """기존 클래스에 문맥 기반 추출기 통합"""
        # 🔥 수정: client가 초기화된 후에만 실행
        if hasattr(self, "client"):
            self.contextual_extractor = ContextualSynonymExtractor(
                client=self.client, verbose=self.verbose
            )
            print("🎯 문맥 기반 정밀 추출기 활성화")
        else:
            print("⚠️ OpenAI 클라이언트가 아직 초기화되지 않음")


class ImprovedSynonymRefiner:
    """개선된 동의어/반의어 정제기 (VOCABULARY.XLSX 구조 출력)"""

    def __init__(
        self,
        vocab_file: str,
        api_key: str = None,
        verbose: bool = False,
        enable_ai_base_form: bool = True,
    ):

        self.vocab_file = vocab_file
        self.verbose = verbose
        self.enable_ai_base_form = enable_ai_base_form

        # 🔥 OpenAI 클라이언트를 먼저 초기화
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API 키가 필요합니다.")

        self.client = openai.OpenAI()  # 한 번만 초기화

        # 🔥 품사 매핑 추가
        self.pos_mapping = {
            "noun": "n",
            "n": "n",
            "NN": "n",
            "NNS": "n",
            "NNP": "n",
            "NNPS": "n",
            "verb": "v",
            "v": "v",
            "VB": "v",
            "VBD": "v",
            "VBG": "v",
            "VBN": "v",
            "VBP": "v",
            "VBZ": "v",
            "adjective": "a",
            "adj": "a",
            "a": "a",
            "JJ": "a",
            "JJR": "a",
            "JJS": "a",
            "adverb": "r",
            "adv": "r",
            "r": "r",
            "RB": "r",
            "RBR": "r",
            "RBS": "r",
        }

        # 데이터 로드
        self.load_vocabulary()

        # 단어장 수준 분석
        self.vocabulary_level = self.analyze_vocabulary_level()

        # API 사용량 추적
        self.gpt_calls = 0
        self.api_calls = {"datamuse": 0, "wordsapi": 0, "merriam": 0}
        self.total_tokens = 0
        self.cost_estimate = 0.0
        self.word_root_cache = {}

        # 캐시
        self.cache = {}
        self.api_cache = {}
        self.base_form_cache = {}
        self.load_cache()

        print("✅ 개선된 동의어/반의어 정제기 초기화 완료")
        print("📚 단어장: {}개 항목".format(len(self.vocab_df)))
        print("🎯 감지된 수준: {}".format(self.vocabulary_level))
        print("🔄 사용 API: Datamuse, WordsAPI, Merriam-Webster")
        if self.enable_ai_base_form:
            print("🤖 AI 원형 판별: 활성화")
        else:
            print("📝 원형 판별: 기존 패턴 방식")

    def _normalize_pos(self, pos: str) -> str:
        """품사를 WordNet 형식으로 정규화"""
        if not pos:
            return ""
        pos_clean = pos.strip().lower()
        return self.pos_mapping.get(pos_clean, pos_clean)

    def _verify_pos_with_wordnet(self, word: str, target_pos: str) -> bool:
        """WordNet으로 단어의 품사 검증"""
        try:
            target_pos_norm = self._normalize_pos(target_pos)
            if not target_pos_norm:
                return True  # 품사 정보가 없으면 통과

            synsets = wordnet.synsets(word, pos=target_pos_norm)
            return len(synsets) > 0
        except:
            return True  # 오류 시 통과

    def _filter_candidates_by_pos(
        self, candidates: List[str], target_pos: str
    ) -> List[str]:
        """품사가 일치하는 후보들만 필터링"""
        if not candidates or not target_pos:
            return candidates

        filtered = []
        for candidate in candidates:
            if self._verify_pos_with_wordnet(candidate, target_pos):
                filtered.append(candidate)
            elif self.verbose:
                print(f"   ⚠️ 품사 불일치로 제외: {candidate} (목표: {target_pos})")

        return filtered

    def load_vocabulary(self):
        """단어장 로드 - VOCABULARY.XLSX 구조 유지"""
        try:
            if self.vocab_file.endswith(".xlsx"):
                self.vocab_df = pd.read_excel(self.vocab_file, engine="openpyxl")
            else:
                self.vocab_df = pd.read_csv(self.vocab_file, encoding="utf-8")

            # VOCABULARY.XLSX의 필수 컬럼들이 있는지 확인
            expected_columns = [
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

            missing_columns = [
                col for col in expected_columns if col not in self.vocab_df.columns
            ]
            if missing_columns:
                print(f"⚠️ 누락된 컬럼들: {missing_columns}")
                print("💡 기본값으로 누락된 컬럼들을 추가합니다.")

                # 누락된 컬럼들을 기본값으로 추가
                for col in missing_columns:
                    if col in ["교재ID"]:
                        self.vocab_df[col] = 1
                    elif col in ["순서"]:
                        self.vocab_df[col] = range(1, len(self.vocab_df) + 1)
                    elif col in ["분리형여부", "사용자DB매칭"]:
                        self.vocab_df[col] = False
                    elif col in ["신뢰도", "동의어신뢰도"]:
                        self.vocab_df[col] = 0.8
                    else:
                        self.vocab_df[col] = ""

            # 동의어/반의어가 있는 항목만 필터링 (개선할 대상)
            mask = (self.vocab_df.get("동의어", "").fillna("").str.strip() != "") | (
                self.vocab_df.get("반의어", "").fillna("").str.strip() != ""
            )

            if mask.sum() == 0:
                print("⚠️ 동의어/반의어가 있는 항목이 없습니다. 모든 항목을 처리합니다.")
                self.process_df = self.vocab_df.copy()
            else:
                self.process_df = self.vocab_df[mask].reset_index(drop=True)
                print("📖 동의어/반의어가 있는 항목: {}개".format(len(self.process_df)))

        except Exception as e:
            raise Exception("단어장 로드 실패: {}".format(e))

    def analyze_vocabulary_level(self) -> str:
        """단어장 전체 수준 분석"""
        if len(self.vocab_df) == 0:
            return "intermediate"

        words = self.vocab_df.get("단어", "").fillna("").str.strip()
        word_lengths = [len(w) for w in words if w]
        avg_length = np.mean(word_lengths) if word_lengths else 5

        complex_patterns = 0
        total_words = 0

        for word in words:
            if not word or len(word) < 2:
                continue
            total_words += 1

            if (
                word.endswith(
                    ("tion", "sion", "ment", "ness", "ity", "ism", "ology", "graphy")
                )
                or word.startswith(
                    ("pre", "post", "anti", "auto", "inter", "trans", "sub", "super")
                )
                or len(word) > 8
            ):
                complex_patterns += 1

        complexity_ratio = complex_patterns / total_words if total_words > 0 else 0

        if avg_length > 7 and complexity_ratio > 0.3:
            return "advanced"
        elif avg_length > 5.5 and complexity_ratio > 0.15:
            return "intermediate"
        else:
            return "elementary"

    def estimate_word_difficulty(self, word: str) -> str:
        """개별 단어 난이도 추정"""
        if not word:
            return "elementary"

        length = len(word)
        score = 0

        if length <= 4:
            score += 1
        elif length <= 6:
            score += 2
        elif length <= 8:
            score += 3
        else:
            score += 4

        if re.search(r"(tion|sion|ment|ness|ity|ism|ology|graphy)$", word):
            score += 2
        if re.search(r"^(pre|post|anti|auto|inter|trans|sub|super)", word):
            score += 1
        if re.search(r"(duct|struct|spect|ject|port|form)", word):
            score += 1

        if score <= 2:
            return "elementary"
        elif score <= 4:
            return "intermediate"
        else:
            return "advanced"

    def check_base_form_batch(self, words: List[str]) -> Dict[str, bool]:
        """AI를 사용해 여러 단어의 원형 여부를 한 번에 확인"""
        if not words or not self.enable_ai_base_form:
            return {word: True for word in words}

        uncached_words = []
        results = {}

        for word in words:
            if word.lower() in self.base_form_cache:
                results[word] = self.base_form_cache[word.lower()]
            else:
                uncached_words.append(word)

        if not uncached_words:
            return results

        # 배치로 AI에게 질의 (최대 20개씩)
        batch_size = 20
        for i in range(0, len(uncached_words), batch_size):
            batch = uncached_words[i : i + batch_size]
            batch_results = self._ai_check_base_forms(batch)
            results.update(batch_results)

            # 캐시에 저장
            for word, is_base in batch_results.items():
                self.base_form_cache[word.lower()] = is_base

        return results

    def _ai_check_base_forms(self, words: List[str]) -> Dict[str, bool]:
        """AI로 단어들의 원형 여부 확인"""
        words_str = ", ".join(words)
        # 🔥 현재 처리 중인 단어의 품사 정보 포함
        current_pos = getattr(self, "_current_pos", "")
        pos_info = f" (목표 품사: {current_pos})" if current_pos else ""

        prompt = f"""
Please analyze the following English words and determine if each word is in its BASE FORM{pos_info}.

Words to analyze: {words_str}

STRICT Rules - Mark as FALSE if:
- Plurals (cats, dogs, children, mice)
- Past tense (walked, ran, went, was) 
- Gerunds/Present participles (running, walking, being, going)
- Past participles (broken, written, done)
- Comparatives/Superlatives (better, best, worse, worst, more, most)
- Any inflected form
{f"- Words that are not {current_pos}" if current_pos else ""}

Mark as TRUE only if:
- The exact dictionary base form{f" that is {current_pos}" if current_pos else ""}
- Root form you would look up in a dictionary

Respond in JSON format only:
{{
    "word1": true/false,
    "word2": true/false,
    ...
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert English linguist who determines if words are in their base/root form.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )

            self.gpt_calls += 1
            if hasattr(response, "usage"):
                usage = response.usage
                self.total_tokens += usage.total_tokens
                cost = (
                    usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60
                ) / 1000000
                self.cost_estimate += cost

            content = response.choices[0].message.content.strip()

            try:
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end != -1:
                        content = content[start:end].strip()

                result = json.loads(content)

                validated_result = {}
                for word in words:
                    if word in result:
                        validated_result[word] = bool(result[word])
                    else:
                        validated_result[word] = False
                        if self.verbose:
                            print(f"⚠️ AI 응답에 '{word}' 없음 - False로 설정")

                if self.verbose:
                    base_count = sum(validated_result.values())
                    print(f"🤖 AI 원형 판별: {len(words)}개 중 {base_count}개가 원형")

                return validated_result

            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"❌ AI 응답 JSON 파싱 실패: {e}")
                return {word: False for word in words}

        except Exception as e:
            if self.verbose:
                print(f"❌ AI 원형 판별 실패: {e}")
            return {word: False for word in words}

    def get_datamuse_synonyms_antonyms(self, word: str) -> Tuple[List[str], List[str]]:
        """Datamuse API에서 동의어/반의어 가져오기"""
        cache_key = f"datamuse_{word.lower()}"
        if cache_key in self.api_cache:
            cached = self.api_cache[cache_key]
            return cached.get("synonyms", []), cached.get("antonyms", [])

        synonyms, antonyms = [], []

        try:
            # 동의어 요청
            syn_data = self._safe_api_call(
                "https://api.datamuse.com/words",
                params={"rel_syn": word, "max": 20},
                timeout=15,
                max_retries=2,
            )
            if syn_data and isinstance(syn_data, list):
                synonyms = [
                    item.get("word", "")
                    for item in syn_data
                    if isinstance(item, dict) and "word" in item
                ]
                synonyms = [w for w in synonyms if w and isinstance(w, str)]

            # 반의어 요청
            ant_data = self._safe_api_call(
                "https://api.datamuse.com/words",
                params={"rel_ant": word, "max": 20},
                timeout=15,
                max_retries=2,
            )
            if ant_data and isinstance(ant_data, list):
                antonyms = [
                    item.get("word", "")
                    for item in ant_data
                    if isinstance(item, dict) and "word" in item
                ]
                antonyms = [w for w in antonyms if w and isinstance(w, str)]

            self.api_calls["datamuse"] += 1
            self.api_cache[cache_key] = {"synonyms": synonyms, "antonyms": antonyms}

            if self.verbose:
                print(
                    f"📊 Datamuse: {word} → 동의어 {len(synonyms)}개, 반의어 {len(antonyms)}개"
                )

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Datamuse API 오류 ({word}): {e}")

        return synonyms or [], antonyms or []

    def get_wordsapi_synonyms_antonyms(self, word: str) -> Tuple[List[str], List[str]]:
        """WordsAPI에서 동의어/반의어 가져오기"""
        cache_key = f"wordsapi_{word.lower()}"
        if cache_key in self.api_cache:
            cached = self.api_cache[cache_key]
            return cached.get("synonyms", []), cached.get("antonyms", [])

        synonyms, antonyms = [], []
        rapidapi_key = os.getenv("RAPIDAPI_KEY")

        if not rapidapi_key:
            if self.verbose:
                print("⚠️ WordsAPI: RAPIDAPI_KEY 환경변수가 설정되지 않음 - 건너뜀")
            return [], []

        try:
            headers = {
                "X-RapidAPI-Key": rapidapi_key,
                "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com",
            }

            # 동의어 요청
            syn_data = self._safe_api_call(
                f"https://wordsapiv1.p.rapidapi.com/words/{word}/synonyms",
                headers=headers,
            )
            if syn_data and isinstance(syn_data, dict) and "synonyms" in syn_data:
                synonyms = syn_data["synonyms"]
                if synonyms and isinstance(synonyms, list):
                    synonyms = [w for w in synonyms if w and isinstance(w, str)]
                else:
                    synonyms = []

            # 반의어 요청
            ant_data = self._safe_api_call(
                f"https://wordsapiv1.p.rapidapi.com/words/{word}/antonyms",
                headers=headers,
            )
            if ant_data and isinstance(ant_data, dict) and "antonyms" in ant_data:
                antonyms = ant_data["antonyms"]
                if antonyms and isinstance(antonyms, list):
                    antonyms = [w for w in antonyms if w and isinstance(w, str)]
                else:
                    antonyms = []

            self.api_calls["wordsapi"] += 1
            self.api_cache[cache_key] = {"synonyms": synonyms, "antonyms": antonyms}

            if self.verbose:
                print(
                    f"📊 WordsAPI: {word} → 동의어 {len(synonyms)}개, 반의어 {len(antonyms)}개"
                )

        except Exception as e:
            if self.verbose:
                print(f"⚠️ WordsAPI 오류 ({word}): {e}")

        return synonyms or [], antonyms or []

    def get_merriam_webster_synonyms_antonyms(
        self, word: str
    ) -> Tuple[List[str], List[str]]:
        """Merriam-Webster Thesaurus API에서 동의어/반의어 가져오기"""
        cache_key = "merriam_{}".format(word.lower())
        if cache_key in self.api_cache:
            return (
                self.api_cache[cache_key]["synonyms"],
                self.api_cache[cache_key]["antonyms"],
            )

        synonyms = []
        antonyms = []

        mw_key = os.getenv("MERRIAM_WEBSTER_KEY")
        if not mw_key:
            if self.verbose:
                print(
                    "⚠️ Merriam-Webster: MERRIAM_WEBSTER_KEY 환경변수가 설정되지 않음 - 건너뜀"
                )
            return synonyms, antonyms

        try:
            url = "https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{}".format(
                word
            )
            params = {"key": mw_key}

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()

                if (
                    isinstance(data, list)
                    and len(data) > 0
                    and isinstance(data[0], dict)
                ):
                    entry = data[0]

                    # 동의어 추출
                    if "meta" in entry and "syns" in entry["meta"]:
                        syns_lists = entry["meta"]["syns"]
                        for syn_list in syns_lists:
                            synonyms.extend(syn_list)

                    # 반의어 추출
                    if "meta" in entry and "ants" in entry["meta"]:
                        ants_lists = entry["meta"]["ants"]
                        for ant_list in ants_lists:
                            antonyms.extend(ant_list)

            self.api_calls["merriam"] += 1
            self.api_cache[cache_key] = {"synonyms": synonyms, "antonyms": antonyms}

            if self.verbose:
                print(
                    "📊 Merriam-Webster: {} → 동의어 {}개, 반의어 {}개".format(
                        word, len(synonyms), len(antonyms)
                    )
                )

        except Exception as e:
            if self.verbose:
                print("⚠️ Merriam-Webster API 오류 ({}): {}".format(word, e))

        return synonyms, antonyms

    def collect_api_candidates(
        self, word: str, pos: str = ""
    ) -> Tuple[List[str], List[str]]:
        """문맥 기반 정밀 후보 수집 (기존 메서드 대체)"""
        if not word or not isinstance(word, str):
            return [], []

        if self.verbose:
            print(f"🔍 {word} 정밀 후보 수집 시작...")

        try:
            # 1단계: 기존 API 결과 수집 (빠른 후보 확보)
            datamuse_syns, datamuse_ants = self.get_datamuse_synonyms_antonyms(word)
            wordsapi_syns, wordsapi_ants = self.get_wordsapi_synonyms_antonyms(word)
            merriam_syns, merriam_ants = self.get_merriam_webster_synonyms_antonyms(
                word
            )

            # 2단계: 기본 필터링
            api_synonyms = self.basic_filter_candidates_for_synonyms(
                datamuse_syns + wordsapi_syns + merriam_syns, word
            )
            api_antonyms = self.basic_filter_candidates_for_synonyms(
                datamuse_ants + wordsapi_ants + merriam_ants, word
            )

            # 3단계: 문맥 기반 정밀 추출 (메인 로직)
            if hasattr(self, "contextual_extractor") and hasattr(
                self, "_current_context"
            ):
                context = getattr(self, "_current_context", "")
                if context:
                    contextual_result = (
                        self.contextual_extractor.extract_contextual_synonyms_antonyms(
                            word, context, pos
                        )
                    )

                    # 문맥 기반 결과 우선 사용
                    final_synonyms = contextual_result["synonyms"]
                    final_antonyms = contextual_result["antonyms"]

                    # API 결과와 병합 (중복 제거)
                    combined_synonyms = list(set(final_synonyms + api_synonyms[:2]))[:3]
                    combined_antonyms = list(set(final_antonyms + api_antonyms[:1]))[:2]

                    if self.verbose:
                        print(
                            f"🎯 문맥 기반 결과: 동의어 {len(final_synonyms)}개, 반의어 {len(final_antonyms)}개"
                        )
                        print(
                            f"🔗 API 병합 후: 동의어 {len(combined_synonyms)}개, 반의어 {len(combined_antonyms)}개"
                        )

                    return combined_synonyms, combined_antonyms

            # 4단계: 문맥 정보가 없으면 기존 방식 (안전장치)
            if self.verbose:
                print("⚠️ 문맥 정보 없음, 기존 API 방식 사용")

            # AI 원형 판별
            if self.enable_ai_base_form and (api_synonyms or api_antonyms):
                all_candidates = list(set(api_synonyms + api_antonyms))
                base_form_results = self.check_base_form_batch(all_candidates)

                api_synonyms = [
                    w for w in api_synonyms if base_form_results.get(w, False)
                ]
                api_antonyms = [
                    w for w in api_antonyms if base_form_results.get(w, False)
                ]

            # 어근 중복 제거 및 개수 제한
            final_synonyms = self.enhanced_filter_synonyms_antonyms(
                api_synonyms, word, max_count=3
            )
            final_antonyms = self.enhanced_filter_synonyms_antonyms(
                api_antonyms, word, max_count=2
            )

            return final_synonyms, final_antonyms

        except Exception as e:
            if self.verbose:
                print(f"❌ 정밀 후보 수집 실패 ({word}): {e}")
            return [], []

    def is_high_quality_candidate(self, candidate: str, original_word: str) -> bool:
        """고품질 후보인지 확인"""
        # 길이가 적당해야 함 (3-10글자)
        if len(candidate) < 3 or len(candidate) > 10:
            return False

        # 일반적인 영어 단어 패턴
        if not re.match(r"^[a-zA-Z]+$", candidate):
            return False

        # 원본과 완전히 다른 단어여야 함
        if (
            candidate.lower() in original_word.lower()
            or original_word.lower() in candidate.lower()
        ):
            return False

        # 매우 일반적인 단어들 (화이트리스트)
        common_words = {
            "big",
            "small",
            "large",
            "tiny",
            "huge",
            "little",
            "good",
            "bad",
            "great",
            "poor",
            "fine",
            "nice",
            "awful",
            "happy",
            "sad",
            "glad",
            "upset",
            "joyful",
            "fast",
            "slow",
            "quick",
            "rapid",
            "hot",
            "cold",
            "warm",
            "cool",
            "easy",
            "hard",
            "difficult",
            "simple",
            "new",
            "old",
            "fresh",
            "recent",
            "important",
            "crucial",
            "vital",
            "significant",
            "beautiful",
            "pretty",
            "lovely",
            "ugly",
            "strong",
            "weak",
            "powerful",
            "rich",
            "poor",
            "wealthy",
            "clean",
            "dirty",
            "messy",
            "loud",
            "quiet",
            "noisy",
            "silent",
        }

        return candidate.lower() in common_words

    def filter_candidates(self, candidates: List[str], original_word: str) -> List[str]:
        """후보 단어들 엄격 필터링"""
        if not candidates or candidates is None:
            return []

        filtered = []
        original_lower = original_word.lower() if original_word else ""

        for candidate in candidates:
            if not candidate or not isinstance(candidate, str):
                continue

            candidate = candidate.strip()
            candidate_lower = candidate.lower()

            # 1. 한 단어만 허용 (공백, 하이픈, 언더스코어 포함된 것 제외)
            if " " in candidate or "-" in candidate or "_" in candidate:
                continue

            # 2. 자기 자신 제외
            if candidate_lower == original_lower:
                continue

            # 3. 너무 짧은 단어 제외 (2글자 이하)
            if len(candidate) <= 2:
                continue

            # 4. 숫자가 포함된 단어 제외
            if any(char.isdigit() for char in candidate):
                continue

            # 5. 특수문자가 포함된 단어 제외 (아포스트로피 제외)
            if re.search(r"[^\w\']", candidate):
                continue

            # 6. 모두 대문자인 단어 제외 (약어일 가능성)
            if candidate.isupper() and len(candidate) > 1:
                continue

            # 7. 원형과 너무 유사한 단어 제외
            if self.is_too_similar(candidate_lower, original_lower):
                continue

            # 8. 일반적이지 않은 단어 제외 (매우 긴 단어)
            if len(candidate) > 12:
                continue

            filtered.append(candidate)

        if filtered:
            # 합성어 제거
            filtered = [w for w in filtered if not self.is_compound_word(w)]
            # 어근 중복 제거
            filtered = self.remove_same_root_duplicates(filtered, original_word)

        return filtered or []

    def is_too_similar(self, word1: str, word2: str) -> bool:
        """두 단어가 너무 유사한지 확인 (단순 edit distance)"""
        if abs(len(word1) - len(word2)) > 3:
            return False

        # 간단한 edit distance (Levenshtein distance)
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        edit_distance = dp[m][n]
        max_len = max(len(word1), len(word2))

        # 편집 거리가 단어 길이의 30% 이하면 너무 유사
        return edit_distance / max_len < 0.3

    def is_compound_word(self, word: str) -> bool:
        """2개 이상 합성어인지 확인"""
        if not word or len(word) < 4:
            return False

        # 명확한 합성어 패턴
        compound_patterns = [
            r"\w+\s+\w+",  # 공백으로 분리된 합성어
            r"\w+-\w+",  # 하이픈으로 연결된 합성어
            r"\w+_\w+",  # 언더스코어로 연결된 합성어
        ]

        for pattern in compound_patterns:
            if re.search(pattern, word):
                return True

        return False

    def extract_word_root(self, word: str) -> str:
        """단어의 어근 추출"""
        if word.lower() in self.word_root_cache:
            return self.word_root_cache[word.lower()]

        original_word = word.lower()

        # 일반적인 접미사 제거 (순서 중요 - 긴 것부터)
        suffixes_to_remove = [
            "ness",
            "ment",
            "tion",
            "sion",
            "ible",
            "able",
            "ical",
            "ful",
            "less",
            "ish",
            "ous",
            "ive",
            "ing",
            "ed",
            "er",
            "est",
            "ly",
            "al",
            "ic",
            "y",
        ]

        # 불규칙 어근 매핑
        irregular_roots = {
            "better": "good",
            "worse": "bad",
            "best": "good",
            "worst": "bad",
            "more": "much",
            "most": "much",
            "less": "little",
            "least": "little",
            "further": "far",
            "furthest": "far",
            "older": "old",
            "oldest": "old",
        }

        if original_word in irregular_roots:
            root = irregular_roots[original_word]
        else:
            root = original_word
            for suffix in suffixes_to_remove:
                if root.endswith(suffix) and len(root) > len(suffix) + 2:
                    potential_root = root[: -len(suffix)]
                    if len(potential_root) >= 3:
                        root = potential_root
                        break

        self.word_root_cache[word.lower()] = root
        return root

    def have_same_root(self, word1: str, word2: str) -> bool:
        """두 단어가 같은 어근을 가지는지 확인"""
        root1 = self.extract_word_root(word1)
        root2 = self.extract_word_root(word2)

        if root1 == root2:
            return True

        if len(root1) >= 4 and len(root2) >= 4:
            if root1 in root2 or root2 in root1:
                return True

        return False

    def remove_same_root_duplicates(
        self, words: List[str], original_word: str
    ) -> List[str]:
        """어근이 같은 단어들 중복 제거"""
        if not words:
            return words

        filtered = []
        seen_roots = set()
        original_root = self.extract_word_root(original_word)

        for word in words:
            if not word:
                continue

            word_root = self.extract_word_root(word)

            # 원본 단어와 같은 어근이면 제외
            if word_root == original_root:
                continue

            # 이미 같은 어근의 단어가 있으면 제외
            if word_root in seen_roots:
                continue

            seen_roots.add(word_root)
            filtered.append(word)

        return filtered

    def _safe_api_call(
        self,
        url: str,
        params: Dict = None,
        headers: Dict = None,
        timeout: int = 10,
        max_retries: int = 3,
    ) -> Optional[List]:
        """안전한 API 호출 (재시도 로직 추가)"""

        for attempt in range(max_retries):
            try:
                current_timeout = timeout + (attempt * 5)
                response = requests.get(
                    url, params=params, headers=headers, timeout=current_timeout
                )
                response.raise_for_status()

                data = response.json()

                if data is None:
                    return []

                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data  # dict 그대로 반환
                else:
                    if self.verbose:
                        print(f"⚠️ 예상치 못한 응답 타입: {type(data)}")
                    return []

            except requests.exceptions.Timeout as e:
                if self.verbose:
                    print(
                        f"⏰ API 호출 시간 초과 ({url}), 시도 {attempt + 1}/{max_retries}: {e}"
                    )
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    if self.verbose:
                        print(f"🔄 {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    if self.verbose:
                        print(f"❌ 최대 재시도 횟수 초과, API 호출 포기: {url}")
                    return []

            except requests.exceptions.RequestException as e:
                if self.verbose:
                    print(
                        f"⚠️ API 호출 실패 ({url}), 시도 {attempt + 1}/{max_retries}: {e}"
                    )
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    if self.verbose:
                        print(f"🔄 {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    return []

            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"⚠️ JSON 파싱 실패 ({url}): {e}")
                return []

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 예상치 못한 오류 ({url}): {e}")
                return []

        return []

    def safe_gpt_call(
        self, messages: List[Dict], max_tokens: int = 400, temperature: float = 0.1
    ) -> Tuple[Optional[str], Optional[str]]:
        """안전한 GPT API 호출"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            self.gpt_calls += 1
            if hasattr(response, "usage"):
                usage = response.usage
                self.total_tokens += usage.total_tokens
                cost = (
                    usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60
                ) / 1000000
                self.cost_estimate += cost

            return response.choices[0].message.content.strip(), None

        except Exception as e:
            return None, str(e)

    def parse_gpt_json(self, content: str) -> Dict:
        """GPT 응답에서 JSON 추출 및 파싱"""
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    content = content[start:end].strip()

            result = json.loads(content)
            return result if isinstance(result, dict) else {}

        except json.JSONDecodeError:
            try:
                lines = content.split("\n")
                partial_result = {}
                for line in lines:
                    if ":" in line and not line.strip().startswith("//"):
                        key_part, value_part = line.split(":", 1)
                        key = key_part.strip().strip('"').strip("'")
                        value = value_part.strip().rstrip(",").strip('"').strip("'")

                        if key in ["final_synonyms", "final_antonyms"]:
                            if "[" in value and "]" in value:
                                try:
                                    import ast

                                    partial_result[key] = ast.literal_eval(value)
                                except:
                                    items = value.strip("[]").split(",")
                                    partial_result[key] = [
                                        item.strip().strip('"').strip("'")
                                        for item in items
                                        if item.strip()
                                    ]
                            else:
                                partial_result[key] = []
                        else:
                            partial_result[key] = value

                return partial_result
            except:
                return {}
        except Exception:
            return {}

    def gpt_validate_and_select(
        self,
        word: str,
        meaning: str,
        context: str,
        api_synonyms: List[str],
        api_antonyms: List[str],
    ) -> Dict:
        """GPT로 문맥 적합성 검증 및 최종 선별 (수정된 버전)"""

        cache_key = "gpt_{}:{}:{}".format(word, context[:50], self.vocabulary_level)
        if cache_key in self.cache:
            return self.cache[cache_key]

        word_level = self.estimate_word_difficulty(word)

        # 🔥 이미 필터링된 후보들 (최대 3개씩)
        api_syns_str = ", ".join(api_synonyms) if api_synonyms else "None"
        api_ants_str = ", ".join(api_antonyms) if api_antonyms else "None"

        # 🔥 품사 정보 추출 및 정규화
        pos = getattr(self, "_current_pos", "")  # 처리 중인 단어의 품사
        pos_normalized = self._normalize_pos(pos)

        pos_description = {
            "n": "noun (명사)",
            "v": "verb (동사)",
            "a": "adjective (형용사)",
            "r": "adverb (부사)",
        }.get(pos_normalized, f"{pos} (기타)")

        prompt = f"""
    You are an English education expert. Apply EXTREMELY STRICT criteria for synonym/antonym validation.

    **Analysis Target:**
    - Word: "{word}"
    - Part of Speech: {pos_description}
    - Meaning: {meaning}
    - Context: {context}
    - Synonym candidates: {api_syns_str}
    - Antonym candidates: {api_ants_str}
    - Vocabulary Level: {self.vocabulary_level}

    **🔥 CRITICAL REQUIREMENTS:**
    1. **EXACT Part of Speech Match:**
    - ALL synonyms and antonyms must be {pos_description} ONLY
    - Do NOT include words from different parts of speech
    - Example: If "{word}" is {pos_description}, provide only {pos_description} synonyms/antonyms

    2. **Single Words ONLY:**
    - NO phrases or multi-word expressions
    - NO hyphenated words
    - NO compound expressions

    3. **Base Forms ONLY:**
    - Reject ALL inflected forms (plurals, past tense, gerunds, comparatives)
    - Accept only dictionary base forms that are {pos_description}

    4. **Perfect Context Match:**
    - Must fit the exact meaning and context
    - Must be appropriate for {self.vocabulary_level} level students

    5. **Conservative Selection:**
    - When in doubt, EXCLUDE rather than include
    - Maximum 2 synonyms, 1 antonym
    - Quality over quantity

    **Response in JSON format:**
    {{
        "final_synonyms": ["max 2 perfect single-word {pos_description} synonyms"],
        "final_antonyms": ["max 1 perfect single-word {pos_description} antonym"],
        "pos_verified": "{pos_normalized}",
        "rejected_synonyms": [
            {{"word": "word", "reason": "specific rejection reason including POS/form issues"}}
        ],
        "rejected_antonyms": [
            {{"word": "word", "reason": "specific rejection reason including POS/form issues"}}
        ],
        "contextual_confidence": 0.0-1.0,
        "reasoning": "detailed explanation including POS verification and selection criteria"
    }}
    """

        messages = [
            {
                "role": "system",
                "content": "You are an ultra-conservative English education expert. You reject most suggestions unless they are absolutely perfect matches. Many words should have NO synonyms or antonyms - this is normal and correct.",
            },
            {"role": "user", "content": prompt},
        ]

        # GPT 호출
        response_content, error = self.safe_gpt_call(
            messages, max_tokens=600, temperature=0.1
        )

        if response_content:
            result = self.parse_gpt_json(response_content)

            # 🔥 추가 안전장치 - 결과를 다시 한번 필터링
            raw_synonyms = result.get("final_synonyms", [])
            raw_antonyms = result.get("final_antonyms", [])

            final_synonyms = self.enhanced_filter_synonyms_antonyms(
                raw_synonyms, word, max_count=2
            )
            final_antonyms = self.enhanced_filter_synonyms_antonyms(
                raw_antonyms, word, max_count=1
            )

            refined_result = {
                "api_synonyms": api_synonyms,
                "api_antonyms": api_antonyms,
                "contextual_meaning": result.get("contextual_meaning", ""),
                "part_of_speech": result.get("part_of_speech", ""),
                "final_synonyms": final_synonyms,  # 🔥 최종 필터링 적용
                "final_antonyms": final_antonyms,  # 🔥 최종 필터링 적용
                "rejected_synonyms": result.get("rejected_synonyms", []),
                "rejected_antonyms": result.get("rejected_antonyms", []),
                "api_confidence": (
                    "high"
                    if len(api_synonyms) >= 2 or len(api_antonyms) >= 2
                    else "medium" if api_synonyms or api_antonyms else "low"
                ),
                "contextual_confidence": result.get("contextual_confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "word_level": word_level,
                "vocab_level": self.vocabulary_level,
                "method": (
                    "hybrid_api_gpt_ai_filtered_enhanced"
                    if self.enable_ai_base_form
                    else "hybrid_api_gpt_enhanced"
                ),
            }

            self.cache[cache_key] = refined_result

            if self.verbose:
                final_syn_count = len(refined_result["final_synonyms"])
                final_ant_count = len(refined_result["final_antonyms"])
                print(
                    f"🤖 GPT 검증: {word} → 최종 동의어 {final_syn_count}개, 반의어 {final_ant_count}개"
                )

            return refined_result

        else:
            if self.verbose:
                print(f"❌ GPT 검증 실패: {word} - {error}")

            # GPT 실패 시에도 필터링 적용
            filtered_synonyms = self.enhanced_filter_synonyms_antonyms(
                api_synonyms, word, max_count=2
            )
            filtered_antonyms = self.enhanced_filter_synonyms_antonyms(
                api_antonyms, word, max_count=1
            )

            return {
                "api_synonyms": api_synonyms,
                "api_antonyms": api_antonyms,
                "contextual_meaning": "",
                "part_of_speech": "",
                "final_synonyms": filtered_synonyms,
                "final_antonyms": filtered_antonyms,
                "rejected_synonyms": [],
                "rejected_antonyms": [],
                "api_confidence": "low",
                "contextual_confidence": 0.3,
                "reasoning": f"GPT 검증 실패, 필터링된 API 결과 사용: {error}",
                "word_level": word_level,
                "vocab_level": self.vocabulary_level,
                "method": "api_only_fallback_enhanced",
            }

    def process_vocabulary(
        self, max_items: int = None, batch_size: int = 10
    ) -> pd.DataFrame:
        """문맥 정보를 포함한 개선된 처리 (기존 메서드 수정)"""

        # 기존 처리 로직과 동일하지만 문맥 정보 추가
        if max_items:
            process_df = self.process_df.head(max_items).copy()
        else:
            process_df = self.process_df.copy()

        total_items = len(process_df)
        processed_count = 0

        print("🎯 문맥 기반 정밀 처리 시작")

        for idx, row in process_df.iterrows():
            word = str(row.get("단어", "")).strip()
            meaning = str(row.get("뜻(한글)", "")).strip()
            context = str(row.get("문맥", "")).strip()
            pos = str(row.get("품사", "")).strip()

            if not word:
                continue

            try:
                start_time = time.time()

                # 🔥 현재 처리 중인 단어의 문맥과 품사 저장
                self._current_pos = pos
                self._current_context = context  # 문맥 정보 추가

                # 1단계: 문맥 기반 정밀 후보 수집
                api_synonyms, api_antonyms = self.collect_api_candidates(word, pos)

                # 2단계: GPT 검증 (기존과 동일)
                result = self.gpt_validate_and_select(
                    word, meaning, context, api_synonyms, api_antonyms
                )

                processing_time = time.time() - start_time

                # 결과 업데이트 (기존과 동일)
                final_synonyms_str = ", ".join(result["final_synonyms"])
                final_antonyms_str = ", ".join(result["final_antonyms"])

                process_df.at[idx, "동의어"] = final_synonyms_str
                process_df.at[idx, "반의어"] = final_antonyms_str

                if result["contextual_meaning"]:
                    process_df.at[idx, "문맥적의미"] = result["contextual_meaning"]

                process_df.at[idx, "동의어신뢰도"] = result["contextual_confidence"]
                process_df.at[idx, "처리방식"] = "문맥기반정밀추출"
                process_df.at[idx, "포함이유"] = result["reasoning"]

                processed_count += 1

                # 진행상황 출력
                if processed_count % batch_size == 0:
                    percentage = (processed_count / total_items) * 100
                    print(
                        f"📈 진행률: {processed_count}/{total_items} ({percentage:.1f}%) - 문맥 기반 정밀 처리"
                    )

            except Exception as e:
                print(f"❌ {word} 처리 실패: {e}")
                continue

        print(f"\n✅ 문맥 기반 정밀 처리 완료: {processed_count}개 항목")
        if max_items:
            process_df = self.process_df.head(max_items).copy()
        else:
            process_df = self.process_df.copy()

        # 임시 컬럼들 추가 (처리용)
        process_df["_원본_동의어"] = process_df["동의어"].copy()
        process_df["_원본_반의어"] = process_df["반의어"].copy()
        process_df["_API_동의어수"] = 0
        process_df["_API_반의어수"] = 0
        process_df["_최종_동의어수"] = 0
        process_df["_최종_반의어수"] = 0
        process_df["_처리_시간"] = ""
        process_df["_개선_방식"] = ""

        method_description = (
            "하이브리드 + AI 원형 필터링" if self.enable_ai_base_form else "하이브리드"
        )
        print("🔄 {} 처리 시작: {}개 항목".format(method_description, total_items))
        print(
            "📊 1단계: API 후보 수집 → 2단계: {} → 3단계: GPT 문맥 검증".format(
                "기존 필터링 + AI 원형 판별"
                if self.enable_ai_base_form
                else "기존 필터링 + 교차 검증"
            )
        )

        processed_count = 0

        for idx, row in process_df.iterrows():
            word = str(row.get("단어", "")).strip()
            meaning = str(row.get("뜻(한글)", "")).strip()
            context = str(row.get("문맥", "")).strip()
            pos = str(row.get("품사", "")).strip()  # 🔥 품사 정보 추가

            if not word:
                continue

            try:
                start_time = time.time()

                # 🔥 현재 처리 중인 단어의 품사 저장 (GPT 프롬프트에서 사용)
                self._current_pos = pos

                # 1단계: API에서 후보 수집 (품사 정보 전달)
                api_synonyms, api_antonyms = self.collect_api_candidates(word, pos)

                # 2단계: GPT로 검증 및 선별 (품사 정보 포함)
                result = self.gpt_validate_and_select(
                    word, meaning, context, api_synonyms, api_antonyms
                )
                processing_time = time.time() - start_time

                # VOCABULARY.XLSX 구조에 맞게 결과 업데이트
                # 기본적으로 원본 데이터 유지하고 동의어/반의어만 개선
                final_synonyms_str = ", ".join(result["final_synonyms"])
                final_antonyms_str = ", ".join(result["final_antonyms"])

                # 동의어/반의어 컬럼 업데이트
                process_df.at[idx, "동의어"] = final_synonyms_str
                process_df.at[idx, "반의어"] = final_antonyms_str

                # 문맥적의미 업데이트 (있는 경우)
                if result["contextual_meaning"]:
                    process_df.at[idx, "문맥적의미"] = result["contextual_meaning"]

                # 동의어신뢰도 업데이트
                process_df.at[idx, "동의어신뢰도"] = result["contextual_confidence"]

                # 처리방식 업데이트
                if self.enable_ai_base_form:
                    process_df.at[idx, "처리방식"] = "개선된정제_AI원형필터링"
                else:
                    process_df.at[idx, "처리방식"] = "개선된정제_하이브리드"

                # 포함이유 업데이트
                process_df.at[idx, "포함이유"] = result["reasoning"]

                # 임시 통계 컬럼들 업데이트
                process_df.at[idx, "_API_동의어수"] = len(result["api_synonyms"])
                process_df.at[idx, "_API_반의어수"] = len(result["api_antonyms"])
                process_df.at[idx, "_최종_동의어수"] = len(result["final_synonyms"])
                process_df.at[idx, "_최종_반의어수"] = len(result["final_antonyms"])
                process_df.at[idx, "_처리_시간"] = "{:.2f}s".format(processing_time)
                process_df.at[idx, "_개선_방식"] = result["method"]

                processed_count += 1

                # 진행상황 출력
                if processed_count % batch_size == 0:
                    percentage = (processed_count / total_items) * 100
                    ai_note = " + AI원형" if self.enable_ai_base_form else ""
                    print(
                        "📈 진행률: {}/{} ({:.1f}%) - API: D{}/W{}/M{}, GPT: {}회{}, 비용: ${:.3f}".format(
                            processed_count,
                            total_items,
                            percentage,
                            self.api_calls["datamuse"],
                            self.api_calls["wordsapi"],
                            self.api_calls["merriam"],
                            self.gpt_calls,
                            ai_note,
                            self.cost_estimate,
                        )
                    )

                # API 제한 고려한 딜레이
                if processed_count % 5 == 0:
                    time.sleep(0.5)

            except Exception as e:
                print("❌ {} 처리 실패: {}".format(word, e))
                continue

        # 최종 통계
        ai_note = " + AI 원형 필터링" if self.enable_ai_base_form else ""
        print("\n✅ 개선된 정제{} 완료!".format(ai_note))
        print("📊 처리된 항목: {}개".format(processed_count))
        print(
            "📡 API 호출: Datamuse {}회, WordsAPI {}회, Merriam {}회".format(
                self.api_calls["datamuse"],
                self.api_calls["wordsapi"],
                self.api_calls["merriam"],
            )
        )
        print(
            "🤖 GPT 호출: {}회{}".format(
                self.gpt_calls, " (원형 판별 포함)" if self.enable_ai_base_form else ""
            )
        )
        print("🎫 총 토큰 사용: {:,}개".format(self.total_tokens))
        print("💰 실제 GPT 비용: ${:.3f}".format(self.cost_estimate))

        if self.enable_ai_base_form:
            print("🤖 AI 원형 판별 캐시: {}개 항목".format(len(self.base_form_cache)))

        # 정제 결과 통계
        self.print_refinement_statistics(process_df)

        return process_df

    def print_refinement_statistics(self, df: pd.DataFrame):
        """정제 결과 통계 출력"""
        ai_note = " + AI 원형 필터링" if self.enable_ai_base_form else ""
        print("\n📈 개선된 정제{} 결과 통계:".format(ai_note))

        # API vs 최종 비교
        api_syn_total = df["_API_동의어수"].sum()
        final_syn_total = df["_최종_동의어수"].sum()
        api_ant_total = df["_API_반의어수"].sum()
        final_ant_total = df["_최종_반의어수"].sum()

        print("🔵 동의어:")
        print(
            "   • API 수집{}: {}개".format(
                " (AI 원형 필터링 후)" if self.enable_ai_base_form else "",
                api_syn_total,
            )
        )
        print("   • GPT 검증 후: {}개".format(final_syn_total))
        if api_syn_total > 0:
            retention_rate = (final_syn_total / api_syn_total) * 100
            print("   • 채택률: {:.1f}%".format(retention_rate))

        print("🔴 반의어:")
        print(
            "   • API 수집{}: {}개".format(
                " (AI 원형 필터링 후)" if self.enable_ai_base_form else "",
                api_ant_total,
            )
        )
        print("   • GPT 검증 후: {}개".format(final_ant_total))
        if api_ant_total > 0:
            retention_rate = (final_ant_total / api_ant_total) * 100
            print("   • 채택률: {:.1f}%".format(retention_rate))

        # 문맥 신뢰도 분포
        confidence_scores = pd.to_numeric(df["동의어신뢰도"], errors="coerce").fillna(0)
        high_conf = (confidence_scores >= 0.7).sum()
        med_conf = ((confidence_scores >= 0.4) & (confidence_scores < 0.7)).sum()
        low_conf = (confidence_scores < 0.4).sum()

        print("📊 문맥 신뢰도 분포:")
        print("   • 높음 (0.7+): {}개".format(high_conf))
        print("   • 보통 (0.4-0.7): {}개".format(med_conf))
        print("   • 낮음 (0.4미만): {}개".format(low_conf))

    def save_results(
        self, df: pd.DataFrame, output_prefix: str = None
    ) -> Dict[str, str]:
        """결과 저장 - VOCABULARY.XLSX와 동일한 구조"""
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.vocab_file))[0]
            ai_suffix = "_ai" if self.enable_ai_base_form else ""
            output_prefix = "{}_improved{}_{}".format(base_name, ai_suffix, timestamp)

        saved_files = {}

        try:
            # VOCABULARY.XLSX와 정확히 동일한 구조로 출력
            expected_columns = [
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

            # 임시 컬럼들 제거하고 원본 구조만 유지
            output_df = df.copy()
            temp_columns = [col for col in output_df.columns if col.startswith("_")]
            output_df = output_df.drop(columns=temp_columns, errors="ignore")

            # 컬럼 순서를 VOCABULARY.XLSX와 동일하게 정렬
            available_columns = [
                col for col in expected_columns if col in output_df.columns
            ]
            output_df = output_df[available_columns]

            # 1. 메인 결과 파일 (VOCABULARY.XLSX와 동일한 구조)
            main_file = "{}_VOCABULARY_IMPROVED.xlsx".format(output_prefix)
            output_df.to_excel(main_file, index=False, engine="openpyxl")
            saved_files["개선된_VOCABULARY파일"] = main_file

            # 2. 상세 분석 결과 (임시 컬럼들 포함)
            detailed_file = "{}_detailed_analysis.xlsx".format(output_prefix)
            df.to_excel(detailed_file, index=False, engine="openpyxl")
            saved_files["상세분석결과"] = detailed_file

            # 3. 비교 분석 버전 (Before/After)
            if all(
                col in df.columns for col in ["단어", "_원본_동의어", "_원본_반의어"]
            ):
                compare_data = []
                for idx, row in df.iterrows():
                    word = row.get("단어", "")
                    original_syn = row.get("_원본_동의어", "")
                    original_ant = row.get("_원본_반의어", "")
                    improved_syn = row.get("동의어", "")
                    improved_ant = row.get("반의어", "")

                    compare_data.append(
                        {
                            "단어": word,
                            "뜻(한글)": row.get("뜻(한글)", ""),
                            "문맥": row.get("문맥", ""),
                            "원본_동의어": original_syn,
                            "개선된_동의어": improved_syn,
                            "원본_반의어": original_ant,
                            "개선된_반의어": improved_ant,
                            "동의어_변화": (
                                "✓ 개선됨" if original_syn != improved_syn else "동일"
                            ),
                            "반의어_변화": (
                                "✓ 개선됨" if original_ant != improved_ant else "동일"
                            ),
                            "신뢰도": row.get("동의어신뢰도", ""),
                            "처리방식": row.get("처리방식", ""),
                            "처리시간": row.get("_처리_시간", ""),
                        }
                    )

                compare_df = pd.DataFrame(compare_data)
                compare_file = "{}_before_after_comparison.xlsx".format(output_prefix)
                compare_df.to_excel(compare_file, index=False, engine="openpyxl")
                saved_files["변화비교"] = compare_file

            # 4. 통계 리포트
            stats_file = "{}_improvement_report.txt".format(output_prefix)
            with open(stats_file, "w", encoding="utf-8") as f:
                ai_note = " + AI 원형 필터링" if self.enable_ai_base_form else ""
                f.write("개선된 동의어/반의어 정제 결과 리포트{}\n".format(ai_note))
                f.write("=" * 70 + "\n")
                f.write(
                    "처리 일시: {}\n".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                f.write("원본 파일: {}\n".format(self.vocab_file))
                f.write("단어장 수준: {}\n".format(self.vocabulary_level))
                f.write("처리된 항목: {}개\n".format(len(df)))
                if self.enable_ai_base_form:
                    f.write("AI 원형 필터링: 활성화\n")
                f.write("\n")

                f.write("🔧 적용된 개선사항:\n")
                f.write("✅ 1. 원형 단어만 허용 (동명사, 복수형, 과거형 등 제외)\n")
                f.write("✅ 2. 억지스러운 반의어 제외 (추상개념, 고유명사 등)\n")
                f.write("✅ 3. 어근 동일한 단어 중복 제거 (order vs orderliness)\n")
                f.write("✅ 4. 2개 이상 합성어 제외 (free person, well-known 등)\n")
                f.write("✅ 5. 동명사, 복수형태 완전 제외\n")
                f.write("✅ 6. VOCABULARY.XLSX와 동일한 구조 유지\n\n")

                f.write("API 사용량:\n")
                f.write("- Datamuse: {}회 (무료)\n".format(self.api_calls["datamuse"]))
                f.write(
                    "- WordsAPI: {}회 (무료 2500/월)\n".format(
                        self.api_calls["wordsapi"]
                    )
                )
                f.write(
                    "- Merriam-Webster: {}회 (무료 1000/일)\n\n".format(
                        self.api_calls["merriam"]
                    )
                )

                f.write("GPT 사용량:\n")
                f.write(
                    "- 호출 횟수: {}회{}\n".format(
                        self.gpt_calls,
                        " (원형 판별 포함)" if self.enable_ai_base_form else "",
                    )
                )
                f.write("- 토큰 사용: {:,}개\n".format(self.total_tokens))
                f.write("- 비용: ${:.3f}\n\n".format(self.cost_estimate))

                if self.enable_ai_base_form:
                    f.write("AI 원형 판별:\n")
                    f.write("- 캐시된 항목: {}개\n\n".format(len(self.base_form_cache)))

                # 상세 통계 추가
                try:
                    api_syn_total = df["_API_동의어수"].sum()
                    final_syn_total = df["_최종_동의어수"].sum()
                    api_ant_total = df["_API_반의어수"].sum()
                    final_ant_total = df["_최종_반의어수"].sum()

                    f.write("처리 결과:\n")
                    f.write(
                        "동의어: API {}개 → 최종 {}개 (채택률 {:.1f}%)\n".format(
                            api_syn_total,
                            final_syn_total,
                            (
                                (final_syn_total / api_syn_total * 100)
                                if api_syn_total > 0
                                else 0
                            ),
                        )
                    )
                    f.write(
                        "반의어: API {}개 → 최종 {}개 (채택률 {:.1f}%)\n".format(
                            api_ant_total,
                            final_ant_total,
                            (
                                (final_ant_total / api_ant_total * 100)
                                if api_ant_total > 0
                                else 0
                            ),
                        )
                    )

                    # 변화 통계
                    if "_원본_동의어" in df.columns:
                        syn_changed = (df["_원본_동의어"] != df["동의어"]).sum()
                        ant_changed = (df["_원본_반의어"] != df["반의어"]).sum()
                        f.write("\n변화 통계:\n")
                        f.write("- 동의어 개선된 항목: {}개\n".format(syn_changed))
                        f.write("- 반의어 개선된 항목: {}개\n".format(ant_changed))
                        f.write(
                            "- 전체 개선율: {:.1f}%\n".format(
                                ((syn_changed + ant_changed) / (len(df) * 2)) * 100
                            )
                        )

                except Exception as e:
                    f.write("통계 계산 오류: {}\n".format(e))

                f.write("\n📁 출력 파일:\n")
                f.write(
                    "- 메인 결과: {}_VOCABULARY_IMPROVED.xlsx (원본과 동일한 구조)\n".format(
                        output_prefix
                    )
                )
                f.write(
                    "- 상세 분석: {}_detailed_analysis.xlsx\n".format(output_prefix)
                )
                f.write(
                    "- 변화 비교: {}_before_after_comparison.xlsx\n".format(
                        output_prefix
                    )
                )

            saved_files["통계리포트"] = stats_file

            # 5. 캐시 저장
            self.save_cache()

            print("\n📁 저장된 파일:")
            for desc, filename in saved_files.items():
                print("   • {}: {}".format(desc, filename))

            print(f"\n🎯 메인 출력 파일: {main_file}")
            print("   → VOCABULARY.XLSX와 동일한 구조로 동의어/반의어가 개선됨")

        except Exception as e:
            print("❌ 파일 저장 실패: {}".format(e))
            if self.verbose:
                import traceback

                traceback.print_exc()

        return saved_files

    def load_cache(self):
        """캐시 로드"""
        # GPT 캐시
        gpt_cache_file = "improved_gpt_cache_{}.json".format(self.vocabulary_level)
        try:
            if os.path.exists(gpt_cache_file):
                with open(gpt_cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print("📦 GPT 캐시 로드: {}개 항목".format(len(self.cache)))
        except Exception as e:
            print("⚠️ GPT 캐시 로드 실패: {}".format(e))
            self.cache = {}

        # API 캐시
        api_cache_file = "improved_api_cache.json"
        try:
            if os.path.exists(api_cache_file):
                with open(api_cache_file, "r", encoding="utf-8") as f:
                    self.api_cache = json.load(f)
                print("📦 API 캐시 로드: {}개 항목".format(len(self.api_cache)))
        except Exception as e:
            print("⚠️ API 캐시 로드 실패: {}".format(e))
            self.api_cache = {}

        # AI 원형 판별 캐시
        if self.enable_ai_base_form:
            base_form_cache_file = "improved_ai_base_form_cache.json"
            try:
                if os.path.exists(base_form_cache_file):
                    with open(base_form_cache_file, "r", encoding="utf-8") as f:
                        self.base_form_cache = json.load(f)
                    print(
                        "📦 AI 원형 판별 캐시 로드: {}개 항목".format(
                            len(self.base_form_cache)
                        )
                    )
            except Exception as e:
                print("⚠️ AI 원형 판별 캐시 로드 실패: {}".format(e))
                self.base_form_cache = {}

        # 어근 캐시 로드
        word_root_cache_file = "improved_word_root_cache.json"
        try:
            if os.path.exists(word_root_cache_file):
                with open(word_root_cache_file, "r", encoding="utf-8") as f:
                    self.word_root_cache = json.load(f)
                print(
                    "📦 어근 분석 캐시 로드: {}개 항목".format(
                        len(self.word_root_cache)
                    )
                )
        except Exception as e:
            print("⚠️ 어근 캐시 로드 실패: {}".format(e))
            self.word_root_cache = {}

    def basic_filter_candidates_for_synonyms(
        self, candidates: List[str], original_word: str
    ) -> List[str]:
        """기본 필터링 - 단일 단어만, 유효한 형태만"""
        if not candidates:
            return []

        filtered = []
        original_lower = original_word.lower() if original_word else ""

        for candidate in candidates:
            if not candidate or not isinstance(candidate, str):
                continue

            candidate = candidate.strip()
            candidate_lower = candidate.lower()

            # 🔥 1. 두 개 이상 단어 제외 (공백, 하이픈, 언더스코어)
            if " " in candidate or "-" in candidate or "_" in candidate:
                continue

            # 2. 자기 자신 제외
            if candidate_lower == original_lower:
                continue

            # 3. 너무 짧은 단어 제외 (2글자 이하)
            if len(candidate) <= 2:
                continue

            # 4. 숫자가 포함된 단어 제외
            if any(char.isdigit() for char in candidate):
                continue

            # 5. 특수문자가 포함된 단어 제외 (아포스트로피 제외)
            if re.search(r"[^\w']", candidate):
                continue

            # 6. 모두 대문자인 단어 제외 (약어일 가능성)
            if candidate.isupper() and len(candidate) > 1:
                continue

            # 7. 너무 긴 단어 제외 (12글자 초과)
            if len(candidate) > 12:
                continue

            # 8. 유효한 영어 알파벳으로만 구성
            if not re.match(r"^[a-zA-Z']+$", candidate):
                continue

            filtered.append(candidate)

        return filtered

    def extract_word_root_for_synonyms(self, word: str) -> str:
        """단어의 어근 추출 (PorterStemmer 사용)"""
        if not word:
            return ""

        # 기존 캐시 확인
        if word.lower() in self.word_root_cache:
            return self.word_root_cache[word.lower()]

        original_word = word.lower()

        # 불규칙 어근 매핑 (주요 불규칙 변화들)
        irregular_roots = {
            "better": "good",
            "best": "good",
            "worse": "bad",
            "worst": "bad",
            "more": "much",
            "most": "much",
            "less": "little",
            "least": "little",
            "further": "far",
            "furthest": "far",
            "farther": "far",
            "farthest": "far",
            "older": "old",
            "oldest": "old",
            "elder": "old",
            "eldest": "old",
        }

        if original_word in irregular_roots:
            root = irregular_roots[original_word]
        else:
            # 간단한 접미사 제거 (순서 중요 - 긴 것부터)
            suffixes = [
                "ness",
                "ment",
                "tion",
                "sion",
                "ible",
                "able",
                "ical",
                "ful",
                "less",
                "ish",
                "ous",
                "ive",
                "ing",
                "ed",
                "er",
                "est",
                "ly",
                "al",
                "ic",
                "y",
            ]

            root = original_word
            for suffix in suffixes:
                if root.endswith(suffix) and len(root) > len(suffix) + 2:
                    potential_root = root[: -len(suffix)]
                    if len(potential_root) >= 3:
                        root = potential_root
                        break

        # 캐시에 저장
        self.word_root_cache[word.lower()] = root
        return root

    def remove_same_root_duplicates_for_synonyms(
        self, words: List[str], original_word: str
    ) -> List[str]:
        """어근이 같은 단어들 중복 제거"""
        if not words:
            return words

        filtered = []
        seen_roots = set()
        original_root = self.extract_word_root_for_synonyms(original_word)

        for word in words:
            if not word:
                continue

            word_root = self.extract_word_root_for_synonyms(word)

            # 원본 단어와 같은 어근이면 제외
            if word_root == original_root:
                continue

            # 이미 같은 어근의 단어가 있으면 제외
            if word_root in seen_roots:
                continue

            # 어근이 너무 짧으면 (2글자 이하) 신뢰할 수 없으므로 단어 자체로 비교
            if len(word_root) <= 2:
                word_for_comparison = word.lower()
                if word_for_comparison in [w.lower() for w in filtered]:
                    continue

            seen_roots.add(word_root)
            filtered.append(word)

        return filtered

    def enhanced_filter_synonyms_antonyms(
        self, candidates: List[str], original_word: str, max_count: int = 3
    ) -> List[str]:
        """
        동의어/반의어 후보들을 엄격하게 필터링

        Args:
            candidates: 후보 단어 리스트
            original_word: 원본 단어
            max_count: 최대 반환 개수

        Returns:
            필터링된 단어 리스트 (최대 max_count개)
        """
        if not candidates:
            return []

        # 1단계: 기본 필터링 (이미 적용되었지만 추가 안전장치)
        filtered = self.basic_filter_candidates_for_synonyms(candidates, original_word)

        # 2단계: 어근 중복 제거
        filtered = self.remove_same_root_duplicates_for_synonyms(
            filtered, original_word
        )

        # 3단계: 최대 개수 제한
        return filtered[:max_count]

    def save_cache(self):
        """캐시 저장"""
        # GPT 캐시 저장
        gpt_cache_file = "improved_gpt_cache_{}.json".format(self.vocabulary_level)
        try:
            with open(gpt_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            if self.verbose:
                print("💾 GPT 캐시 저장: {}개 항목".format(len(self.cache)))
        except Exception as e:
            print("⚠️ GPT 캐시 저장 실패: {}".format(e))

        # API 캐시 저장
        api_cache_file = "improved_api_cache.json"
        try:
            with open(api_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.api_cache, f, ensure_ascii=False, indent=2)
            if self.verbose:
                print("💾 API 캐시 저장: {}개 항목".format(len(self.api_cache)))
        except Exception as e:
            print("⚠️ API 캐시 저장 실패: {}".format(e))

        # AI 원형 판별 캐시 저장
        if self.enable_ai_base_form:
            base_form_cache_file = "improved_ai_base_form_cache.json"
            try:
                with open(base_form_cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.base_form_cache, f, ensure_ascii=False, indent=2)
                if self.verbose:
                    print(
                        "💾 AI 원형 판별 캐시 저장: {}개 항목".format(
                            len(self.base_form_cache)
                        )
                    )
            except Exception as e:
                print("⚠️ AI 원형 판별 캐시 저장 실패: {}".format(e))

        # 어근 캐시 저장
        word_root_cache_file = "improved_word_root_cache.json"
        try:
            with open(word_root_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.word_root_cache, f, ensure_ascii=False, indent=2)
            if self.verbose:
                print(
                    "💾 어근 분석 캐시 저장: {}개 항목".format(
                        len(self.word_root_cache)
                    )
                )
        except Exception as e:
            print("⚠️ 어근 캐시 저장 실패: {}".format(e))


def main():
    """메인 함수"""
    if not check_dependencies():
        return
    parser = argparse.ArgumentParser(
        description="개선된 동의어/반의어 정제 시스템 (VOCABULARY.XLSX 구조 출력)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
python improved_synonym_refiner.py vocabulary.xlsx
python improved_synonym_refiner.py vocab.xlsx --max-items 50
python improved_synonym_refiner.py vocab.xlsx --api-key your_openai_key
python improved_synonym_refiner.py vocab.xlsx --verbose --output-prefix my_improved
python improved_synonym_refiner.py vocab.xlsx --disable-ai-base-form  # AI 원형 판별 비활성화

새로운 기능:
- VOCABULARY.XLSX와 정확히 동일한 구조로 출력
- AI 기반 원형 단어 판별 (기본 활성화)
- running, cats, better 등 변형된 형태 자동 제거
- 기존 모든 기능과 성능 유지
- Before/After 비교 분석 제공

API 키 설정 (선택사항):
export RAPIDAPI_KEY="your_rapidapi_key"          # WordsAPI용
export MERRIAM_WEBSTER_KEY="your_merriam_key"    # Merriam-Webster용

※ API 키가 없어도 Datamuse API만으로 동작합니다 (무료, 무제한)

출력 파일:
- {파일명}_VOCABULARY_IMPROVED.xlsx : 메인 결과 (VOCABULARY.XLSX와 동일한 구조)
- {파일명}_detailed_analysis.xlsx : 상세 분석 결과
- {파일명}_before_after_comparison.xlsx : 변화 비교
- {파일명}_improvement_report.txt : 통계 리포트
    """,
    )

    parser.add_argument("vocab_file", help="처리할 VOCABULARY.XLSX 파일")
    parser.add_argument(
        "--api-key", help="OpenAI API 키 (환경변수 OPENAI_API_KEY도 사용 가능)"
    )
    parser.add_argument("--max-items", type=int, help="최대 처리 항목 수 (비용 제한)")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="배치 크기 (기본값: 10)"
    )
    parser.add_argument("--output-prefix", help="출력 파일 접두사")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 출력")
    parser.add_argument(
        "--disable-ai-base-form",
        action="store_true",
        help="AI 원형 판별 비활성화 (기존 방식만 사용)",
    )

    args = parser.parse_args()

    try:
        # 파일 존재 확인
        if not os.path.exists(args.vocab_file):
            print("❌ 파일을 찾을 수 없습니다: {}".format(args.vocab_file))
            return

        # API 키 상태 확인
        print("🔑 API 키 상태 확인:")
        print(
            "   • OpenAI: {}".format(
                "설정됨" if (args.api_key or os.getenv("OPENAI_API_KEY")) else "❌ 필수"
            )
        )
        print(
            "   • RapidAPI (WordsAPI): {}".format(
                "설정됨" if os.getenv("RAPIDAPI_KEY") else "선택사항"
            )
        )
        print(
            "   • Merriam-Webster: {}".format(
                "설정됨" if os.getenv("MERRIAM_WEBSTER_KEY") else "선택사항"
            )
        )
        print("   • Datamuse: 무료 (키 불요)")

        if not (args.api_key or os.getenv("OPENAI_API_KEY")):
            print("❌ OpenAI API 키가 필요합니다.")
            return

        # AI 원형 판별 설정
        enable_ai_base_form = not args.disable_ai_base_form
        if enable_ai_base_form:
            print("🤖 AI 원형 판별: 활성화 (--disable-ai-base-form으로 비활성화 가능)")
        else:
            print("📝 AI 원형 판별: 비활성화 (기존 패턴 방식만 사용)")

        print("📋 출력 구조: VOCABULARY.XLSX와 동일한 22개 컬럼 구조 유지")

        # 정제기 초기화
        refiner = ImprovedSynonymRefiner(
            vocab_file=args.vocab_file,
            api_key=args.api_key,
            verbose=args.verbose,
            enable_ai_base_form=enable_ai_base_form,
        )

        # 처리 실행
        result_df = refiner.process_vocabulary(
            max_items=args.max_items, batch_size=args.batch_size
        )

        # 결과 저장
        saved_files = refiner.save_results(result_df, args.output_prefix)

        ai_note = " + AI 원형 판별" if enable_ai_base_form else ""
        print("\n🎉 개선된 정제{} 완료!".format(ai_note))
        print("📋 VOCABULARY.XLSX와 동일한 구조로 출력되었습니다!")
        if enable_ai_base_form:
            print("🤖 AI 원형 판별로 변형된 단어들이 제거되었습니다!")

        # 메인 출력 파일 강조
        main_file = saved_files.get("개선된_VOCABULARY파일", "")
        if main_file:
            print(f"\n🎯 메인 결과 파일: {main_file}")
            print("   → 이 파일을 기존 VOCABULARY.XLSX 대신 사용하세요!")

        print("\n💡 더 많은 API를 사용하려면 환경변수를 설정하세요:")
        print("   export RAPIDAPI_KEY='your_key'")
        print("   export MERRIAM_WEBSTER_KEY='your_key'")

    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print("❌ 오류 발생: {}".format(e))
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
