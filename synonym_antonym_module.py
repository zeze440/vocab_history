"""
동의어/반의어 추출 모듈 (Synonym & Antonym Extractor)
마지막 버전에 추가할 수 있는 독립적인 모듈

사용법:
1. 기존 코드에 이 모듈을 추가
2. AdvancedVocabExtractor 클래스에 통합
3. 단어장 생성 시 동의어/반의어 자동 추출
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
import openai
import re
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer


class SynonymAntonymExtractor:
    """독립적인 동의어/반의어 추출기"""

    def __init__(self, client, cache_file="synonym_antonym_cache.json", verbose=False):
        self.client = client
        self.cache_file = cache_file
        self.verbose = verbose
        self.cache = {}
        self.gpt_calls = 0
        self.max_gpt_calls = 500  # GPT 호출 제한

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
        # 기본 단어 세트 (제외할 단어들)
        self.basic_words = {
            "good",
            "bad",
            "big",
            "small",
            "make",
            "take",
            "get",
            "go",
            "come",
            "see",
            "hear",
            "feel",
            "know",
            "think",
            "say",
            "tell",
            "have",
            "be",
            "do",
            "can",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
        }

        self._load_cache()

    def _load_cache(self):
        """캐시 로드"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"✅ 동의어/반의어 캐시 로드: {len(self.cache)}개 항목")
        except Exception as e:
            print(f"⚠️ 캐시 로드 실패: {e}")
            self.cache = {}

    def _save_cache(self):
        """캐시 저장"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            if self.verbose:
                print(f"✅ 동의어/반의어 캐시 저장: {len(self.cache)}개 항목")
        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")

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

    def extract_synonyms_antonyms(
        self, word: str, context: str = "", pos: str = "", meaning: str = ""
    ) -> Dict[str, Any]:
        """
        단어의 동의어/반의어 추출

        Args:
            word: 분석할 단어
            context: 문맥
            pos: 품사
            meaning: 한글 의미

        Returns:
            동의어/반의어 정보 딕셔너리
        """

        # 🔥 품사 정규화
        pos_normalized = self._normalize_pos(pos)

        # 캐시 키에 품사 정보 포함
        cache_key = f"{word.lower()}:{context[:50]}:{pos_normalized}"

        # 안전한 기본값
        safe_default = {
            "synonyms": [],
            "antonyms": [],
            "contextual_meaning": meaning or "",
            "confidence": 0.0,
            "source": "default",
            "notes": "기본값",
        }

        # 캐시 키 생성
        cache_key = f"{word.lower()}:{context[:50]}:{pos}"

        # 캐시 확인
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if isinstance(cached_result, dict):
                return cached_result
            else:
                return safe_default

        # GPT 호출 제한 확인
        if self.gpt_calls >= self.max_gpt_calls:
            if self.verbose:
                print(f"   ⚠️ GPT 호출 한도 도달, WordNet 사용: {word}")
            result = self._get_wordnet_synonyms_antonyms(word)
            self.cache[cache_key] = result
            return result

        # GPT 기반 추출 시도
        try:
            result = self._extract_with_gpt(word, context, pos_normalized, meaning)
            # 🔥 여기를 enhanced 버전으로 변경
            validated_result = self._validate_gpt_result_enhanced(
                result, word, pos_normalized
            )
            validated_result["source"] = "gpt"
            self.cache[cache_key] = validated_result
            return validated_result
        except Exception as e:
            if self.verbose:
                print(f"   ❌ GPT 추출 실패 ({word}): {e}")

            # 🔥 WordNet fallback도 enhanced 버전으로 변경
            result = self._get_wordnet_synonyms_antonyms_enhanced(word, pos_normalized)
            result["notes"] = f"GPT 실패, WordNet 사용: {str(e)}"
            self.cache[cache_key] = result
            return result

    def _extract_with_gpt(
        self, word: str, context: str, pos: str, meaning: str
    ) -> Dict[str, Any]:
        """GPT를 활용한 동의어/반의어 추출"""

        # 🔥 품사 정보 강화
        pos_normalized = self._normalize_pos(pos)
        pos_description = {
            "n": "noun (명사)",
            "v": "verb (동사)",
            "a": "adjective (형용사)",
            "r": "adverb (부사)",
        }.get(pos_normalized, f"{pos} (기타)")

        prompt = f"""
Analyze the English word "{word}" in the given context and provide PRECISE synonyms and antonyms.

Word: "{word}"
Context: "{context}"
Korean meaning: "{meaning}"
Part of speech: {pos_description}

🎯 CRITICAL FILTERING REQUIREMENTS:
1. ONLY provide words that have the SAME meaning as "{word}" in THIS specific context
2. Do NOT include words from different contexts or meanings of "{word}"
3. ONLY single words (no phrases, no hyphens, no underscores)
4. Maximum 3 synonyms and 2 antonyms
5. Ensure contextual relevance - each word must fit naturally in the given context

Example of BAD filtering:
- "bank" in financial context → Do NOT include "shore, riverbank" (different meaning)
- "light" as adjective → Do NOT include "illumination, lamp" (different part of speech)

Example of GOOD filtering:
- "significant" in academic context → "important, substantial, considerable" ✓
- "analyze" in research context → "examine, investigate, study" ✓

For "{word}" in context "{context}":
Provide ONLY words that can replace "{word}" in THIS specific sentence without changing the meaning.

Respond in JSON format:
{{
    "synonyms": ["word1", "word2", "word3"],
    "antonyms": ["word1", "word2"],
    "contextual_meaning": "specific meaning in this context",
    "confidence": 0.8,
    "pos_verified": "{pos_normalized}",
    "notes": "reasoning for each choice"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 비용 절약
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert English linguist specializing in contextual word analysis. Provide accurate synonyms and antonyms based on specific contextual usage.",
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

            # 결과 검증 및 정리
            validated_result = self._validate_gpt_result(result, word)
            validated_result["source"] = "gpt"

            if self.verbose:
                syns = len(validated_result.get("synonyms", []))
                ants = len(validated_result.get("antonyms", []))
                print(f"   📚 GPT 추출: {word} → 동의어 {syns}개, 반의어 {ants}개")

            return validated_result

        except json.JSONDecodeError as e:
            raise Exception(f"JSON 파싱 실패: {e}")
        except Exception as e:
            raise Exception(f"GPT 호출 실패: {e}")

    def _deduplicate_by_stem(self, words: List[str]) -> List[str]:
        """어간(stem)이 중복되는 단어는 한 개만 남기고 제거"""
        ps = PorterStemmer()
        seen_stems = set()
        deduped = []

        for word in words:
            stem = ps.stem(word.lower())
            if stem not in seen_stems:
                deduped.append(word)
                seen_stems.add(stem)
            elif self.verbose:
                print(f"   ⚠️ 어근 중복 제거됨: {word} (stem: {stem})")

        return deduped

    def _validate_gpt_result(self, result: Dict, word: str) -> Dict[str, Any]:
        """GPT 결과 검증 및 정리 (기본 버전)"""

        validated = {
            "synonyms": [],
            "antonyms": [],
            "contextual_meaning": str(result.get("contextual_meaning", "")),
            "confidence": min(max(float(result.get("confidence", 0.5)), 0.0), 1.0),
            "notes": str(result.get("notes", "")),
        }

        # 동의어 검증
        synonyms_raw = result.get("synonyms", [])
        if isinstance(synonyms_raw, list):
            valid_synonyms = []
            for syn in synonyms_raw:
                if isinstance(syn, str) and self._is_valid_single_word(syn, word):
                    valid_synonyms.append(syn.strip())

            # 🔥 어근 중복 제거 및 최대 3개 제한
            validated["synonyms"] = self._enhanced_filter_with_root_dedup(
                valid_synonyms, word, max_count=3
            )

        # 반의어 검증
        antonyms_raw = result.get("antonyms", [])
        if isinstance(antonyms_raw, list):
            valid_antonyms = []
            for ant in antonyms_raw:
                if isinstance(ant, str) and self._is_valid_single_word(ant, word):
                    valid_antonyms.append(ant.strip())

            # 🔥 어근 중복 제거 및 최대 2개 제한
            validated["antonyms"] = self._enhanced_filter_with_root_dedup(
                valid_antonyms, word, max_count=2
            )

        return validated

    def _is_valid_single_word(self, candidate: str, original_word: str) -> bool:
        """단일 단어 유효성 검사"""
        candidate = candidate.strip()

        # 두 개 이상 단어 차단
        if " " in candidate or "-" in candidate or "_" in candidate:
            return False

        # 숫자나 특수문자 포함 차단 (아포스트로피 제외)
        if not re.match(r"^[a-zA-Z']+$", candidate):
            return False

        # 길이 제한
        if len(candidate) < 3 or len(candidate) > 12:
            return False

        # 원본과 동일한 것 제거
        if candidate.lower() == original_word.lower():
            return False

        # 기본 단어 제거
        if candidate.lower() in self.basic_words:
            return False

        return True

    def _enhanced_filter_with_root_dedup(
        self, candidates: List[str], original_word: str, max_count: int = 3
    ) -> List[str]:
        """어근 중복 제거 + 개수 제한 필터링"""

        if not candidates:
            return []

        try:
            from nltk.stem import PorterStemmer

            ps = PorterStemmer()

            seen_stems = set()

            # 원본 단어의 어근 먼저 추가
            original_stem = ps.stem(original_word.lower())
            seen_stems.add(original_stem)

            filtered_results = []

            for candidate in candidates:
                candidate_stem = ps.stem(candidate.lower())

                # 어근이 중복되지 않은 경우만 추가
                if candidate_stem not in seen_stems:
                    filtered_results.append(candidate)
                    seen_stems.add(candidate_stem)

                    # 최대 개수 제한
                    if len(filtered_results) >= max_count:
                        break

            if self.verbose and len(candidates) > len(filtered_results):
                removed_count = len(candidates) - len(filtered_results)
                print(
                    f"   🔄 어근 중복 제거: {removed_count}개 제거, {len(filtered_results)}개 유지"
                )

            return filtered_results

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ 어근 중복 제거 실패: {e}, 기본 중복 제거 사용")

            # NLTK 오류 시 기본 중복 제거 + 개수 제한
            unique_candidates = list(dict.fromkeys(candidates))
            return unique_candidates[:max_count]

    def _get_wordnet_synonyms_antonyms_enhanced(
        self, word: str, pos: str = ""
    ) -> Dict[str, Any]:
        """WordNet 기반 동의어/반의어 추출 (향상된 버전 - 최대 3개 제한)"""

        try:
            synonyms = []
            antonyms = []

            # WordNet에서 동의어 집합 가져오기
            pos_normalized = self._normalize_pos(pos)
            if pos_normalized:
                synsets = wordnet.synsets(word, pos=pos_normalized)
            else:
                synsets = wordnet.synsets(word)

            for synset in synsets:
                # 동의어 수집
                for lemma in synset.lemmas():
                    if lemma.name().lower() != word.lower():
                        synonym = lemma.name().replace("_", " ")
                        # 🔥 즉시 필터링 적용
                        if self._is_valid_single_word(synonym, word):
                            synonyms.append(synonym)

                # 반의어 수집
                for lemma in synset.lemmas():
                    for antonym in lemma.antonyms():
                        antonym_word = antonym.name().replace("_", " ")
                        # 🔥 즉시 필터링 적용
                        if self._is_valid_single_word(antonym_word, word):
                            antonyms.append(antonym_word)

            # 🔥 어근 중복 제거 및 개수 제한
            filtered_synonyms = self._enhanced_filter_with_root_dedup(
                synonyms, word, max_count=3
            )
            filtered_antonyms = self._enhanced_filter_with_root_dedup(
                antonyms, word, max_count=2
            )

            return {
                "synonyms": filtered_synonyms,
                "antonyms": filtered_antonyms,
                "contextual_meaning": "",
                "confidence": 0.6,
                "source": "wordnet_enhanced",
                "notes": "WordNet 기반 추출 (최대 3개 제한, 어근 중복 제거)",
            }

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ WordNet 추출 실패 ({word}): {e}")

            return {
                "synonyms": [],
                "antonyms": [],
                "contextual_meaning": "",
                "confidence": 0.0,
                "source": "failed",
                "notes": f"WordNet 추출 실패: {str(e)}",
            }

    def batch_extract(
        self, word_list: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        여러 단어의 동의어/반의어를 배치로 추출

        Args:
            word_list: [{"word": "단어", "context": "문맥", "pos": "품사", "meaning": "의미"}, ...]

        Returns:
            {단어: 동의어/반의어 정보} 딕셔너리
        """
        results = {}

        print(f"🔍 동의어/반의어 배치 추출 시작: {len(word_list)}개 단어")

        for i, word_info in enumerate(word_list):
            word = word_info.get("word", "")
            context = word_info.get("context", "")
            pos = word_info.get("pos", "")
            meaning = word_info.get("meaning", "")

            if not word:
                continue

            try:
                result = self.extract_synonyms_antonyms(word, context, pos, meaning)
                results[word] = result

                # 진행률 표시
                if (i + 1) % 10 == 0:
                    print(f"   📊 진행률: {i + 1}/{len(word_list)} 완료")

                # 캐시 주기적 저장 (50개마다)
                if (i + 1) % 50 == 0:
                    self._save_cache()

            except Exception as e:
                if self.verbose:
                    print(f"   ❌ {word} 처리 실패: {e}")
                results[word] = {
                    "synonyms": [],
                    "antonyms": [],
                    "contextual_meaning": "",
                    "confidence": 0.0,
                    "source": "error",
                    "notes": f"처리 실패: {str(e)}",
                }

        # 최종 캐시 저장
        self._save_cache()

        # 통계 출력
        successful = len([r for r in results.values() if r.get("confidence", 0) > 0])
        with_synonyms = len(
            [r for r in results.values() if len(r.get("synonyms", [])) > 0]
        )
        with_antonyms = len(
            [r for r in results.values() if len(r.get("antonyms", [])) > 0]
        )

        print(f"✅ 배치 추출 완료:")
        print(f"   📊 성공: {successful}/{len(word_list)}개")
        print(f"   📊 동의어 포함: {with_synonyms}개")
        print(f"   📊 반의어 포함: {with_antonyms}개")
        print(f"   📊 GPT 호출: {self.gpt_calls}회")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """추출 통계 반환"""
        total_items = len(self.cache)

        if total_items == 0:
            return {"total_items": 0}

        with_synonyms = 0
        with_antonyms = 0
        total_synonyms = 0
        total_antonyms = 0
        gpt_count = 0
        wordnet_count = 0

        for result in self.cache.values():
            if isinstance(result, dict):
                synonyms = result.get("synonyms", [])
                antonyms = result.get("antonyms", [])

                if synonyms:
                    with_synonyms += 1
                    total_synonyms += len(synonyms)

                if antonyms:
                    with_antonyms += 1
                    total_antonyms += len(antonyms)

                source = result.get("source", "unknown")
                if source == "gpt":
                    gpt_count += 1
                elif source == "wordnet":
                    wordnet_count += 1

        return {
            "total_items": total_items,
            "with_synonyms": with_synonyms,
            "with_antonyms": with_antonyms,
            "synonym_coverage": (
                (with_synonyms / total_items * 100) if total_items > 0 else 0
            ),
            "antonym_coverage": (
                (with_antonyms / total_items * 100) if total_items > 0 else 0
            ),
            "avg_synonyms": (
                (total_synonyms / with_synonyms) if with_synonyms > 0 else 0
            ),
            "avg_antonyms": (
                (total_antonyms / with_antonyms) if with_antonyms > 0 else 0
            ),
            "gpt_count": gpt_count,
            "wordnet_count": wordnet_count,
            "gpt_calls_used": self.gpt_calls,
        }


# ======= 기존 AdvancedVocabExtractor에 통합하는 코드 =======


def integrate_synonym_extractor_to_vocab_extractor():
    """
    기존 AdvancedVocabExtractor 클래스에 동의어/반의어 기능을 통합하는 방법

    다음 코드를 AdvancedVocabExtractor 클래스에 추가하세요:
    """

    integration_code = '''
# AdvancedVocabExtractor 클래스의 __init__ 메서드에 추가:
def __init__(self, ...):
    # ... 기존 코드 ...
    
    # 🔥 동의어/반의어 추출기 초기화 (새로 추가)
    try:
        self.synonym_extractor = SynonymAntonymExtractor(
            client=client, 
            cache_file="synonym_antonym_cache.json",
            verbose=self.verbose
        )
        print("✅ 동의어/반의어 추출기 초기화 완료")
    except Exception as e:
        print(f"⚠️ 동의어/반의어 추출기 초기화 실패: {e}")
        self.synonym_extractor = None

# AdvancedVocabExtractor 클래스에 새 메서드 추가:
def add_synonyms_antonyms_to_results(self, results):
    """결과에 동의어/반의어 정보 추가"""
    if not hasattr(self, 'synonym_extractor') or not self.synonym_extractor:
        print("⚠️ 동의어/반의어 추출기가 없습니다")
        return results
    
    print("🔍 동의어/반의어 추출 중...")
    
    # 단어 정보 준비
    word_list = []
    for result in results:
        word_info = {
            "word": result.get("original", result.get("단어", "")),
            "context": result.get("context", result.get("문맥", "")),
            "pos": result.get("pos", result.get("품사", "")),
            "meaning": result.get("korean_meaning", result.get("뜻(한글)", ""))
        }
        word_list.append(word_info)
    
    # 배치 추출
    synonym_results = self.synonym_extractor.batch_extract(word_list)
    
    # 결과에 동의어/반의어 정보 추가
    for result in results:
        word = result.get("original", result.get("단어", ""))
        if word in synonym_results:
            syn_data = synonym_results[word]
            result["synonyms"] = ", ".join(syn_data.get("synonyms", []))
            result["antonyms"] = ", ".join(syn_data.get("antonyms", []))
            result["동의어"] = result["synonyms"]  # 한글 컬럼명도 지원
            result["반의어"] = result["antonyms"]
            result["synonym_confidence"] = syn_data.get("confidence", 0.0)
    
    return results

# process_text 메서드 수정 (마지막 부분에 추가):
def process_text(self, ...):
    # ... 기존 코드 ...
    
    # 🔥 동의어/반의어 추가 (새로 추가)
    if rows:  # 결과가 있을 때만
        try:
            rows = self.add_synonyms_antonyms_to_results(rows)
            print(f"✅ 동의어/반의어 추가 완료: {len(rows)}개 항목")
        except Exception as e:
            print(f"⚠️ 동의어/반의어 추가 실패: {e}")
    
    return rows
'''

    return integration_code


# ======= 독립 실행 예제 =======


def example_usage():
    """사용 예제"""

    # OpenAI 클라이언트 초기화 (실제 사용 시 필요)
    # client = openai.OpenAI()

    # 동의어/반의어 추출기 생성
    # extractor = SynonymAntonymExtractor(client, verbose=True)

    # 단일 단어 추출
    # result = extractor.extract_synonyms_antonyms(
    #     word="significant",
    #     context="This is a significant achievement in the field of science.",
    #     pos="adjective",
    #     meaning="중요한, 의미 있는"
    # )
    # print("동의어:", result["synonyms"])
    # print("반의어:", result["antonyms"])

    # 배치 추출
    # word_list = [
    #     {"word": "significant", "context": "significant impact", "pos": "adj", "meaning": "중요한"},
    #     {"word": "analyze", "context": "analyze the data", "pos": "verb", "meaning": "분석하다"},
    #     {"word": "complex", "context": "complex problem", "pos": "adj", "meaning": "복잡한"}
    # ]
    # results = extractor.batch_extract(word_list)

    # 통계 확인
    # stats = extractor.get_statistics()
    # print("추출 통계:", stats)

    pass


if __name__ == "__main__":
    print("동의어/반의어 추출 모듈")
    print("이 모듈을 기존 vocab_extractor에 통합하세요.")
    print("\n통합 방법:")
    print("1. 이 파일을 같은 디렉토리에 저장")
    print(
        "2. vocab_extractor.py에서 'from synonym_antonym_module import SynonymAntonymExtractor' import"
    )
    print("3. AdvancedVocabExtractor 클래스에 위의 통합 코드 추가")
