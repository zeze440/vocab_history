# difficulty_analyzer.py - 난이도 분석 모듈

import json
from nltk.corpus import wordnet
from package_manager import get_nlp_model
from config import BASIC_VERBS, BASIC_ADJECTIVES, BASIC_NOUNS
from external_vocab_db import is_basic_by_external_db


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
        from utils import load_json_safe

        self.difficulty_cache = load_json_safe(self.cache_file, {})
        if self.difficulty_cache:
            print(f"✅ 난이도 필터 캐시 로드: {len(self.difficulty_cache)}개")

    def _save_cache(self):
        """캐시 저장"""
        from utils import save_json_safe

        save_json_safe(self.difficulty_cache, self.cache_file)

    def _analyze_user_db_baseline(self):
        """사용자 DB 단어들의 평균 난이도 분석하여 기준점 설정"""
        if not self.user_words or len(self.user_words) == 0:
            return

        print("🔍 사용자 DB 단어 난이도 기준점 분석 중...")

        # 사용자 DB에서 단일 단어만 선택하여 샘플링 (숙어 제외)
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

        # 사용자 DB에 있는 단어는 무조건 적합 (최우선)
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
- Proper nouns: names of people, places, countries, organizations, brands
- Capitalized words that are names or locations
- Any word that is primarily a proper noun (even if used as common noun)

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

{baseline_info}

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

        # 기본 단어 강력 차단
        if analysis.get("is_basic_vocabulary", False):
            return False, "기본어휘제외"

        difficulty_score = analysis.get("difficulty_score", 5)

        # 대폭 상향된 임계값 (기존 4.0 → 6.0)
        MIN_DIFFICULTY = 7.0

        if difficulty_score < MIN_DIFFICULTY:
            return False, f"난이도부족({difficulty_score}<{MIN_DIFFICULTY})"

        # GPT 추천 확인
        recommendation = analysis.get("recommendation", "exclude")
        if recommendation == "include":
            return True, f"고난이도확인({difficulty_score}점)"
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
