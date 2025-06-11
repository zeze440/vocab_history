# quality_checker.py - 숙어-단어 충돌 검사 + 한글 뜻 및 동의어 보완 기능 추가

import pandas as pd
import os
import re
from collections import defaultdict
import json


class ContextualVocabularyQualityChecker:
    def __init__(
        self, vocabulary_file, openai_client, cache_file="quality_check_cache.json"
    ):
        self.vocabulary_file = vocabulary_file
        self.client = openai_client
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.ai_call_count = 0

        # DataFrame 로드
        self.df = (
            pd.read_excel(vocabulary_file)
            if vocabulary_file.endswith(".xlsx")
            else pd.read_csv(vocabulary_file)
        )

        # 컬럼명 매핑
        self.column_mapping = {
            "word": ["단어", "원형", "word", "base_form", "original"],
            "meaning": ["뜻(한글)", "의미", "한글뜻", "meaning", "korean_meaning"],
            "context": ["문맥", "예문", "context", "sentence", "example"],
            "passage_id": ["지문ID", "passage_id", "text_id", "source_id"],
            "original_text": ["원문", "original", "found_text"],
            "type": ["유형", "type", "entry_type", "item_type"],
            "is_idiom": ["숙어여부", "is_idiom", "is_phrase", "idiom_flag"],
            "synonyms": ["동의어", "synonyms", "synonym"],
            "antonyms": ["반의어", "antonyms", "antonym"],
        }

        self.actual_columns = self._find_actual_columns()
        print(f"🔍 종합 품질 검사기 초기화 완료 (한글 뜻 및 동의어 보완 포함)")

    def _find_actual_columns(self):
        """실제 컬럼명 찾기"""
        actual = {}
        df_columns_lower = [col.lower() for col in self.df.columns]

        for standard_name, possible_names in self.column_mapping.items():
            for possible in possible_names:
                if possible.lower() in df_columns_lower:
                    actual_col = self.df.columns[
                        df_columns_lower.index(possible.lower())
                    ]
                    actual[standard_name] = actual_col
                    break

        return actual

    def load_cache(self):
        """캐시 로드"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_cache(self):
        """캐시 저장"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")

    def generate_contextual_quality_report(self):
        """종합 문맥 품질 보고서 생성"""
        print("🔍 종합 문맥 품질 분석 시작...")

        results = {
            "total_entries": len(self.df),
            "ai_calls_used": 0,
            "issues_found": [],
            "quality_breakdown": {
                "within_passage_duplicates": 0,
                "context_meaning_mismatches": 0,
                "word_meaning_mismatches": 0,
                "idiom_word_conflicts": 0,
                "missing_korean_meanings": 0,  # 🔥 새로 추가
                "missing_synonyms": 0,  # 🔥 새로 추가
                "total_issues": 0,
            },
        }

        # 1. 지문 내 중복 검사
        print("📝 지문 내 중복 단어 검사...")
        duplicate_issues = self._check_within_passage_duplicates()
        results["issues_found"].extend(duplicate_issues)
        results["quality_breakdown"]["within_passage_duplicates"] = len(
            duplicate_issues
        )

        # 2. 숙어-단어 충돌 검사
        print("🔗 숙어-단어 충돌 검사...")
        conflict_issues = self._check_idiom_word_conflicts()
        results["issues_found"].extend(conflict_issues)
        results["quality_breakdown"]["idiom_word_conflicts"] = len(conflict_issues)

        # 3. 🔥 한글 뜻 누락/오류 검사 (새로 추가)
        print("🇰🇷 한글 뜻 품질 검사...")
        korean_meaning_issues = self._check_korean_meaning_quality()
        results["issues_found"].extend(korean_meaning_issues)
        results["quality_breakdown"]["missing_korean_meanings"] = len(korean_meaning_issues)

        # 4. 🔥 동의어/반의어 누락 검사 (새로 추가)
        print("🔗 동의어/반의어 품질 검사...")
        synonym_issues = self._check_synonym_quality()
        results["issues_found"].extend(synonym_issues)
        results["quality_breakdown"]["missing_synonyms"] = len(synonym_issues)

        # 5. 문맥-의미 적합성 검사 (AI 사용)
        print("🤖 문맥-의미 적합성 검사...")
        context_issues = self._check_context_meaning_alignment()
        results["issues_found"].extend(context_issues)
        results["quality_breakdown"]["context_meaning_mismatches"] = len(context_issues)

        # 6. 단어-의미 정확성 검사 (AI 사용)
        print("🔍 단어-의미 정확성 검사...")
        word_meaning_issues = self._check_word_meaning_accuracy()
        results["issues_found"].extend(word_meaning_issues)
        results["quality_breakdown"]["word_meaning_mismatches"] = len(
            word_meaning_issues
        )

        # 총 문제 수 계산
        results["quality_breakdown"]["total_issues"] = len(results["issues_found"])
        results["ai_calls_used"] = self.ai_call_count

        # 품질 점수 계산
        results["quality_score"] = self._calculate_contextual_quality_score(results)

        # 캐시 저장
        self.save_cache()

        return results

    def _check_korean_meaning_quality(self):
        """한글 뜻 품질 검사"""
        issues = []

        if "meaning" not in self.actual_columns or "word" not in self.actual_columns:
            print("⚠️ 한글 뜻 검사를 위한 필수 컬럼을 찾을 수 없음")
            return issues

        meaning_col = self.actual_columns["meaning"]
        word_col = self.actual_columns["word"]

        for idx, row in self.df.iterrows():
            word = str(row[word_col]).strip()
            meaning = str(row[meaning_col]).strip()

            # 문제 패턴 감지
            is_problematic = False
            problem_type = ""

            # 1. "단어의 의미" 패턴
            if meaning.endswith("의 의미") or meaning.endswith("의미"):
                is_problematic = True
                problem_type = "generic_meaning_pattern"

            # 2. 영어가 포함된 경우
            elif any(char.isalpha() and ord(char) < 128 for char in meaning):
                is_problematic = True
                problem_type = "contains_english"

            # 3. 비어있거나 너무 짧은 경우
            elif len(meaning) < 2 or meaning in ["", "nan", "None"]:
                is_problematic = True
                problem_type = "missing_or_too_short"

            # 4. 너무 긴 경우 (30자 초과)
            elif len(meaning) > 30:
                is_problematic = True
                problem_type = "too_long"

            if is_problematic:
                issues.append({
                    "type": "missing_korean_meaning",
                    "severity": "high" if problem_type in ["missing_or_too_short", "generic_meaning_pattern"] else "medium",
                    "index": idx,
                    "word": word,
                    "current_meaning": meaning,
                    "problem_type": problem_type,
                    "description": f"'{word}'의 한글 뜻이 부적절함: {meaning}",
                    "recommendation": "AI로 올바른 한글 뜻 생성 필요"
                })

        print(f"   📊 한글 뜻 문제 발견: {len(issues)}개")
        return issues

    def _check_synonym_quality(self):
        """동의어/반의어 품질 검사"""
        issues = []

        if "synonyms" not in self.actual_columns or "word" not in self.actual_columns:
            print("⚠️ 동의어 검사를 위한 컬럼을 찾을 수 없음 (선택사항)")
            return issues

        synonyms_col = self.actual_columns["synonyms"]
        word_col = self.actual_columns["word"]

        empty_synonyms = 0
        for idx, row in self.df.iterrows():
            word = str(row[word_col]).strip()
            synonyms = str(row[synonyms_col]).strip()

            # 동의어가 비어있는 경우
            if not synonyms or synonyms in ["", "nan", "None", "NaN"]:
                empty_synonyms += 1
                issues.append({
                    "type": "missing_synonyms",
                    "severity": "low",
                    "index": idx,
                    "word": word,
                    "description": f"'{word}'의 동의어가 비어있음",
                    "recommendation": "AI로 동의어 생성 권장"
                })

        print(f"   📊 동의어 누락: {empty_synonyms}개")
        return issues

    def _check_idiom_word_conflicts(self):
        """숙어-단어 충돌 검사"""
        issues = []

        required_cols = ["word", "passage_id"]
        if not all(col in self.actual_columns for col in required_cols):
            print("⚠️ 숙어-단어 충돌 검사를 위한 필수 컬럼을 찾을 수 없음")
            return issues

        word_col = self.actual_columns["word"]
        passage_col = self.actual_columns["passage_id"]
        type_col = self.actual_columns.get("type", None)

        # 숙어와 단어 구분
        idioms = set()
        single_words = set()

        # 지문별로 그룹화
        for passage_id, group in self.df.groupby(passage_col):
            passage_idioms = set()
            passage_words = set()

            for _, row in group.iterrows():
                word = str(row[word_col]).strip().lower()
                entry_type = (
                    str(row[type_col]).lower()
                    if type_col and pd.notna(row[type_col])
                    else ""
                )

                # 숙어 판별: 공백이 있거나 type에 idiom/phrase 포함
                if (
                    " " in word
                    or "idiom" in entry_type
                    or "phrase" in entry_type
                    or "숙어" in entry_type
                ):
                    passage_idioms.add(word)
                    idioms.add(word)
                else:
                    passage_words.add(word)
                    single_words.add(word)

            # 이 지문에서 충돌 검사
            conflicts = self._find_idiom_word_conflicts_in_passage(
                passage_id, passage_idioms, passage_words, group
            )
            issues.extend(conflicts)

        print(f"   📊 전체 숙어: {len(idioms)}개, 단일 단어: {len(single_words)}개")
        print(f"   📊 숙어-단어 충돌 발견: {len(issues)}개")

        return issues

    def _find_idiom_word_conflicts_in_passage(
        self, passage_id, idioms, words, group_df
    ):
        """특정 지문에서 숙어-단어 충돌 찾기"""
        conflicts = []

        word_col = self.actual_columns["word"]

        for idiom in idioms:
            # 숙어를 구성하는 단어들 추출
            idiom_words = set(idiom.split())

            # 이 지문의 단일 단어들과 비교
            conflicting_words = idiom_words.intersection(words)

            if conflicting_words:
                for conflicting_word in conflicting_words:
                    # 충돌하는 단어의 인덱스들 찾기
                    word_indices = group_df[
                        group_df[word_col].str.lower() == conflicting_word
                    ].index.tolist()
                    idiom_indices = group_df[
                        group_df[word_col].str.lower() == idiom
                    ].index.tolist()

                    conflicts.append(
                        {
                            "type": "idiom_word_conflict",
                            "severity": "high",
                            "passage_id": passage_id,
                            "idiom": idiom,
                            "conflicting_word": conflicting_word,
                            "word_indices": word_indices,
                            "idiom_indices": idiom_indices,
                            "description": f"지문 {passage_id}에서 숙어 '{idiom}'과 단일 단어 '{conflicting_word}'가 중복 추출됨",
                            "recommendation": f"숙어 '{idiom}'이 있으므로 단일 단어 '{conflicting_word}' 제거 권장",
                        }
                    )

        return conflicts

    def _check_within_passage_duplicates(self):
        """지문 내 중복 단어 검사 (기존 코드)"""
        issues = []

        if "passage_id" not in self.actual_columns or "word" not in self.actual_columns:
            print("⚠️ 지문ID 또는 단어 컬럼을 찾을 수 없음")
            return issues

        passage_col = self.actual_columns["passage_id"]
        word_col = self.actual_columns["word"]

        # 지문별로 그룹화
        passage_groups = self.df.groupby(passage_col)

        for passage_id, group in passage_groups:
            if len(group) <= 1:
                continue

            # 각 지문 내에서 중복 단어 찾기
            word_counts = group[word_col].value_counts()
            duplicates = word_counts[word_counts > 1]

            for word, count in duplicates.items():
                duplicate_indices = group[group[word_col] == word].index.tolist()
                issues.append(
                    {
                        "type": "within_passage_duplicate",
                        "severity": "high",
                        "passage_id": passage_id,
                        "word": word,
                        "count": count,
                        "indices": duplicate_indices,
                        "description": f"지문 {passage_id}에서 '{word}' 단어가 {count}번 중복됨",
                    }
                )

        print(f"   📊 지문 내 중복 발견: {len(issues)}개")
        return issues

    def _check_context_meaning_alignment(self):
        """문맥-의미 적합성 검사 (기존 코드 유지)"""
        issues = []

        required_cols = ["word", "meaning", "context"]
        if not all(col in self.actual_columns for col in required_cols):
            print("⚠️ 필수 컬럼(단어, 의미, 문맥)을 찾을 수 없음")
            return issues

        word_col = self.actual_columns["word"]
        meaning_col = self.actual_columns["meaning"]
        context_col = self.actual_columns["context"]

        # 샘플링
        check_df = self.df.dropna(subset=[word_col, meaning_col, context_col])
        sample_size = min(50, len(check_df))  # 최대 50개만 검사
        check_df = check_df.sample(n=sample_size, random_state=42)

        print(f"   🤖 {len(check_df)}개 항목의 문맥-의미 적합성 검사 중...")

        for idx, row in check_df.iterrows():
            word = str(row[word_col]).strip()
            meaning = str(row[meaning_col]).strip()
            context = str(row[context_col]).strip()

            # 캐시 확인
            cache_key = f"context_meaning_{word}_{hash(context)}_{hash(meaning)}"
            if cache_key in self.cache:
                result = self.cache[cache_key]
            else:
                result = self._check_context_meaning_with_ai(word, meaning, context)
                if result:
                    self.cache[cache_key] = result
                    self.ai_call_count += 1

            if result and not result.get("is_appropriate", True):
                issues.append(
                    {
                        "type": "context_meaning_mismatch",
                        "severity": result.get("severity", "medium"),
                        "index": idx,
                        "word": word,
                        "meaning": meaning,
                        "context": (
                            context[:100] + "..." if len(context) > 100 else context
                        ),
                        "ai_analysis": result.get("explanation", ""),
                        "description": f"'{word}'의 한글 뜻 '{meaning}'이 문맥에 부적절함",
                    }
                )

        print(f"   📊 문맥-의미 부적합 발견: {len(issues)}개")
        return issues

    def _check_word_meaning_accuracy(self):
        """단어-의미 정확성 검사 (기존 코드 유지)"""
        issues = []
        
        required_cols = ["word", "meaning"]
        if not all(col in self.actual_columns for col in required_cols):
            print("⚠️ 필수 컬럼(단어, 의미)을 찾을 수 없음")
            return issues

        word_col = self.actual_columns["word"]
        meaning_col = self.actual_columns["meaning"]

        check_df = self.df.dropna(subset=[word_col, meaning_col])
        sample_size = min(30, len(check_df))  # 최대 30개만 검사
        check_df = check_df.sample(n=sample_size, random_state=42)

        print(f"   🤖 {len(check_df)}개 항목의 단어-의미 정확성 검사 중...")

        for idx, row in check_df.iterrows():
            word = str(row[word_col]).strip()
            meaning = str(row[meaning_col]).strip()

            # 캐시 확인
            cache_key = f"word_meaning_{word}_{hash(meaning)}"
            if cache_key in self.cache:
                result = self.cache[cache_key]
            else:
                result = self._check_word_meaning_with_ai(word, meaning)
                if result:
                    self.cache[cache_key] = result
                    self.ai_call_count += 1

            if result and not result.get("is_accurate", True):
                issues.append(
                    {
                        "type": "word_meaning_mismatch",
                        "severity": result.get("severity", "medium"),
                        "index": idx,
                        "word": word,
                        "meaning": meaning,
                        "ai_analysis": result.get("explanation", ""),
                        "suggested_meaning": result.get("suggested_meaning", ""),
                        "description": f"'{word}'의 한글 뜻 '{meaning}'이 원형 단어의 의미와 불일치",
                    }
                )

        print(f"   📊 단어-의미 불일치 발견: {len(issues)}개")
        return issues

    def _check_context_meaning_with_ai(self, word, meaning, context):
        """AI를 사용한 문맥-의미 적합성 검사"""
        prompt_template = """Please evaluate whether the provided Korean meaning is appropriate for the English word in the given context.

**Word**: WORD_PLACEHOLDER
**Korean Meaning**: MEANING_PLACEHOLDER  
**Context**: CONTEXT_PLACEHOLDER

Evaluation Criteria:
1. Does the Korean meaning fit the word's usage in this context?
2. Is it contextually appropriate?

Please respond in the following JSON format:
{
    "is_appropriate": true,
    "severity": "low",
    "explanation": "Reason for the evaluation"
}"""
        
        prompt = (prompt_template
                 .replace("WORD_PLACEHOLDER", word)
                 .replace("MEANING_PLACEHOLDER", meaning)
                 .replace("CONTEXT_PLACEHOLDER", context[:200]))

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()
            
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                is_appropriate = ("true" in result_text.lower() and 
                                "is_appropriate" in result_text.lower())
                return {
                    "is_appropriate": is_appropriate,
                    "severity": "medium", 
                    "explanation": result_text,
                }

        except Exception as e:
            print(f"⚠️ AI 문맥-의미 검사 실패 ({word}): {e}")
            return None

    def _check_word_meaning_with_ai(self, word, meaning):
        """AI를 사용한 단어-의미 정확성 검사"""
        prompt_template = """Please evaluate whether the provided Korean meaning is appropriate for the English word.

**Word**: WORD_PLACEHOLDER
**Provided Korean Meaning**: MEANING_PLACEHOLDER

Evaluation Criteria:
1. Does the Korean meaning match the word's basic dictionary meaning?
2. Is it accurate and appropriate?
3. Is it neither too general nor too specific?

Please respond in the following JSON format:
{
    "is_accurate": true,
    "severity": "low",
    "explanation": "Reason why it's inappropriate or appropriate",
    "suggested_meaning": "More appropriate Korean meaning (only if inappropriate)"
}"""

        prompt = (prompt_template
                 .replace("WORD_PLACEHOLDER", word)
                 .replace("MEANING_PLACEHOLDER", meaning))

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()

            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                is_accurate = (
                    "true" in result_text.lower()
                    and "is_accurate" in result_text.lower()
                )
                return {
                    "is_accurate": is_accurate,
                    "severity": "medium",
                    "explanation": result_text,
                }

        except Exception as e:
            print(f"⚠️ AI 단어-의미 검사 실패 ({word}): {e}")
            return None

    def fix_quality_issues(self, apply_fixes=True):
        """품질 문제 자동 수정 (한글 뜻 및 동의어 보완 포함)"""
        if not apply_fixes:
            print("⚠️ 자동 수정이 비활성화됨")
            return self.df

        print("🔧 종합 품질 문제 자동 수정 시작...")
        fixed_df = self.df.copy()
        fix_count = 0

        # 1. 🔥 한글 뜻 보완 (최우선)
        korean_fixes = self._fix_korean_meanings(fixed_df)
        fixed_df = korean_fixes["df"]
        fix_count += korean_fixes["fixed_count"]

        # 2. 🔥 동의어/반의어 보완
        synonym_fixes = self._fix_synonyms_antonyms(fixed_df)
        fixed_df = synonym_fixes["df"]
        fix_count += synonym_fixes["fixed_count"]

        # 3. 숙어-단어 충돌 해결
        conflict_fixes = self._fix_idiom_word_conflicts(fixed_df)
        fixed_df = conflict_fixes["df"]
        fix_count += conflict_fixes["fixed_count"]

        # 4. 지문 내 중복 제거
        duplicate_fixes = self._fix_within_passage_duplicates(fixed_df)
        fixed_df = duplicate_fixes["df"]
        fix_count += duplicate_fixes["fixed_count"]

        print(f"🎯 총 {fix_count}개 항목 수정 완료")
        return fixed_df

    def _fix_korean_meanings(self, df):
        """한글 뜻 자동 보완"""
        print("🇰🇷 한글 뜻 자동 보완 중...")

        if "meaning" not in self.actual_columns or "word" not in self.actual_columns:
            return {"df": df, "fixed_count": 0}

        fixed_df = df.copy()
        fix_count = 0
        meaning_col = self.actual_columns["meaning"]
        word_col = self.actual_columns["word"]
        context_col = self.actual_columns.get("context", None)

        for idx, row in fixed_df.iterrows():
            word = str(row[word_col]).strip()
            meaning = str(row[meaning_col]).strip()
            context = str(row[context_col]).strip() if context_col else ""

            # 문제 있는 한글 뜻 감지
            needs_fix = (
                meaning.endswith("의 의미") or 
                meaning.endswith("의미") or
                any(char.isalpha() and ord(char) < 128 for char in meaning) or
                len(meaning) < 2 or 
                meaning in ["", "nan", "None", "NaN"]
            )

            if needs_fix:
                # AI로 한글 뜻 생성
                new_meaning = self._generate_korean_meaning_with_ai(word, context)
                if new_meaning and new_meaning != f"{word}의 의미":
                    fixed_df.at[idx, meaning_col] = new_meaning
                    fix_count += 1
                    print(f"   ✅ '{word}': '{meaning}' → '{new_meaning}'")

        print(f"   📊 한글 뜻 수정: {fix_count}개")
        return {"df": fixed_df, "fixed_count": fix_count}

    def _generate_korean_meaning_with_ai(self, word, context=""):
        """AI를 사용한 한글 뜻 생성"""
        # 캐시 확인
        cache_key = f"korean_meaning_{word}_{hash(context[:100])}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 숙어인지 단어인지 판별
        is_phrase = " " in word

        try:
            if is_phrase:
                prompt = f"""다음 영어 숙어/구문의 한국어 뜻을 간단명료하게 제시해주세요.

**숙어/구문**: {word}
**문맥**: {context[:200]}

조건:
1. 한국어로만 답변 (영어 단어 포함 금지)
2. 간단명료하게 (5단어 이내)
3. 문맥에 맞는 의미
4. "~하다", "~되다" 등 동사형으로 끝나는 것이 좋음

예시:
- "give up" → "포기하다"
- "look forward to" → "기대하다"
- "a lot of" → "많은"

한국어 뜻:"""
            else:
                prompt = f"""다음 영어 단어의 한국어 뜻을 간단명료하게 제시해주세요.

**단어**: {word}
**문맥**: {context[:200]}

조건:
1. 한국어로만 답변 (영어 단어 포함 금지)
2. 간단명료하게 (3단어 이내)
3. 문맥에 맞는 의미
4. 기본 사전적 의미 위주

예시:
- "efficient" → "효율적인"
- "remarkable" → "놀라운"
- "analysis" → "분석"

한국어 뜻:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip()
            
            # 결과 검증 및 정리
            korean_meaning = self._clean_korean_meaning(result, word)
            
            # 캐시 저장
            self.cache[cache_key] = korean_meaning
            self.ai_call_count += 1
            
            return korean_meaning

        except Exception as e:
            print(f"⚠️ 한글 뜻 생성 실패 ({word}): {e}")
            return f"{word}의 의미"

    def _clean_korean_meaning(self, result, word):
        """한글 뜻 결과 정리"""
        # 불필요한 문구 제거
        result = result.replace("한국어 뜻:", "").strip()
        result = result.replace("답변:", "").strip()
        result = result.replace("의미:", "").strip()
        
        # 따옴표 제거
        result = result.strip('"\'')
        
        # 영어 단어가 포함된 경우 처리
        if any(char.isalpha() and ord(char) < 128 for char in result):
            # 영어가 포함되어 있으면 기본값 반환
            return f"{word}의 의미"
        
        # 너무 길거나 비어있는 경우
        if len(result) > 20 or len(result) < 1:
            return f"{word}의 의미"
        
        return result

    def _generate_synonyms_with_ai(self, word, context=""):
        """AI를 사용한 동의어/반의어 생성"""
        # 캐시 확인
        cache_key = f"synonyms_{word}_{hash(context[:100])}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return cached.get('synonyms', ''), cached.get('antonyms', '')

        try:
            prompt = f"""다음 영어 단어의 동의어와 반의어를 찾아주세요.

**단어**: {word}
**문맥**: {context[:200]}

조건:
1. 동의어 3개 이내 (쉼표로 구분)
2. 반의어 2개 이내 (쉼표로 구분)
3. 문맥에 맞는 단어들만
4. 너무 어려운 단어는 제외
5. 없으면 빈 문자열로 답변

형식:
동의어: word1, word2, word3
반의어: word1, word2

예시:
**단어**: happy
동의어: glad, joyful, cheerful
반의어: sad, unhappy"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip()
            
            # 결과 파싱
            synonyms, antonyms = self._parse_synonyms_result(result)
            
            # 캐시 저장
            self.cache[cache_key] = {
                'synonyms': synonyms,
                'antonyms': antonyms
            }
            self.ai_call_count += 1
            
            return synonyms, antonyms

        except Exception as e:
            print(f"⚠️ 동의어/반의어 생성 실패 ({word}): {e}")
            return "", ""

    def _parse_synonyms_result(self, result):
        """동의어/반의어 결과 파싱"""
        synonyms = ""
        antonyms = ""
        
        lines = result.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('동의어:') or line.startswith('Synonyms:'):
                synonyms = line.split(':', 1)[1].strip()
            elif line.startswith('반의어:') or line.startswith('Antonyms:'):
                antonyms = line.split(':', 1)[1].strip()
        
        # 정리
        synonyms = synonyms.replace(' ', '').strip() if synonyms else ""
        antonyms = antonyms.replace(' ', '').strip() if antonyms else ""
        
        return synonyms, antonyms

    def _fix_idiom_word_conflicts(self, df):
        """숙어-단어 충돌 자동 수정"""
        print("🔗 숙어-단어 충돌 자동 수정 중...")

        fixed_df = df.copy()
        fix_count = 0

        required_cols = ["word", "passage_id"]
        if not all(col in self.actual_columns for col in required_cols):
            return {"df": fixed_df, "fixed_count": 0}

        word_col = self.actual_columns["word"]
        passage_col = self.actual_columns["passage_id"]
        type_col = self.actual_columns.get("type", None)

        indices_to_remove = set()

        # 지문별로 처리
        for passage_id, group in fixed_df.groupby(passage_col):
            passage_idioms = []
            passage_single_words = []

            # 숙어와 단일 단어 분류
            for idx, row in group.iterrows():
                word = str(row[word_col]).strip().lower()
                entry_type = (
                    str(row[type_col]).lower()
                    if type_col and pd.notna(row[type_col])
                    else ""
                )

                if (
                    " " in word
                    or "idiom" in entry_type
                    or "phrase" in entry_type
                    or "숙어" in entry_type
                ):
                    passage_idioms.append((idx, word))
                else:
                    passage_single_words.append((idx, word))

            # 충돌 해결: 숙어가 있으면 해당 단일 단어 제거
            for idiom_idx, idiom in passage_idioms:
                idiom_words = set(idiom.split())

                for word_idx, single_word in passage_single_words:
                    if single_word in idiom_words:
                        indices_to_remove.add(word_idx)
                        fix_count += 1
                        print(f"   ✅ 제거: '{single_word}' (숙어 '{idiom}' 때문에)")

        # 충돌하는 단일 단어들 제거
        if indices_to_remove:
            fixed_df = fixed_df.drop(indices_to_remove).reset_index(drop=True)

        print(f"   📊 숙어-단어 충돌 해결: {fix_count}개 단일 단어 제거")
        return {"df": fixed_df, "fixed_count": fix_count}

    def _fix_within_passage_duplicates(self, df):
        """지문 내 중복 제거 (기존 코드 개선)"""
        print("📝 지문 내 중복 제거 중...")

        fixed_df = df.copy()
        fix_count = 0

        if "passage_id" not in self.actual_columns or "word" not in self.actual_columns:
            return {"df": fixed_df, "fixed_count": 0}

        passage_col = self.actual_columns["passage_id"]
        word_col = self.actual_columns["word"]
        context_col = self.actual_columns.get("context", None)

        groups_to_keep = []

        for passage_id, group in fixed_df.groupby(passage_col):
            for word, word_group in group.groupby(word_col):
                if len(word_group) > 1:
                    # 가장 좋은 것 선택
                    if context_col and context_col in fixed_df.columns:
                        # 문맥 길이로 판단
                        best_idx = (
                            word_group[context_col].astype(str).str.len().idxmax()
                        )
                        groups_to_keep.append(word_group.loc[[best_idx]])
                        fix_count += len(word_group) - 1
                    else:
                        # 첫 번째 것만 유지
                        groups_to_keep.append(word_group.iloc[[0]])
                        fix_count += len(word_group) - 1
                else:
                    groups_to_keep.append(word_group)

        if groups_to_keep:
            fixed_df = pd.concat(groups_to_keep, ignore_index=True)

        print(f"   📊 지문 내 중복 제거: {fix_count}개")
        return {"df": fixed_df, "fixed_count": fix_count}

    def _calculate_contextual_quality_score(self, results):
        """종합 문맥 기반 품질 점수 계산"""
        total_entries = results["total_entries"]
        if total_entries == 0:
            return 0

        breakdown = results["quality_breakdown"]

        score = 100

        # 지문 내 중복 (25점)
        duplicate_penalty = (
            breakdown["within_passage_duplicates"] / total_entries
        ) * 25
        score -= duplicate_penalty

        # 숙어-단어 충돌 (20점)
        conflict_penalty = (breakdown["idiom_word_conflicts"] / total_entries) * 20
        score -= conflict_penalty

        # 🔥 한글 뜻 누락/오류 (25점) - 새로 추가
        korean_meaning_penalty = (breakdown["missing_korean_meanings"] / total_entries) * 25
        score -= korean_meaning_penalty

        # 🔥 동의어 누락 (10점) - 새로 추가  
        synonym_penalty = (breakdown["missing_synonyms"] / total_entries) * 10
        score -= synonym_penalty

        # 문맥-의미 부적합 (15점)
        context_penalty = (breakdown["context_meaning_mismatches"] / total_entries) * 15
        score -= context_penalty

        # 단어-의미 부정확 (5점)
        word_penalty = (breakdown["word_meaning_mismatches"] / total_entries) * 5
        score -= word_penalty

        return round(max(0, score), 1)
    def _fix_synonyms_antonyms(self, df):
        """문맥적으로 정확한 동의어/반의어만 보완"""
        print("🔗 정확한 동의어/반의어 보완 중...")
        
        fixed_df = df.copy()
        fix_count = 0
        
        # 필요한 컬럼 확인
        if "word" not in self.actual_columns:
            return {"df": fixed_df, "fixed_count": 0}
        
        word_col = self.actual_columns["word"]
        meaning_col = self.actual_columns.get("meaning", None)
        context_col = self.actual_columns.get("context", None)
        synonyms_col = self.actual_columns.get("synonyms", None)
        antonyms_col = self.actual_columns.get("antonyms", None)
        
        # 동의어 컬럼이 없으면 건너뜀
        if not synonyms_col:
            print("   💡 동의어 컬럼 없음 - 건너뜀")
            return {"df": fixed_df, "fixed_count": 0}
        
        # 동의어가 비어있거나 부실한 항목들 찾기
        candidates = []
        for idx, row in fixed_df.iterrows():
            word = str(row[word_col]).strip()
            synonyms = str(row[synonyms_col]).strip() if pd.notna(row[synonyms_col]) else ""
            
            # 숙어는 건너뜀
            if " " in word:
                continue
                
            # 동의어가 비어있거나 의미없는 경우만 처리
            if (not synonyms or 
                synonyms in ["", "nan", "NaN", "None"] or
                len(synonyms.split(',')) < 2):  # 동의어가 1개 이하인 경우
                candidates.append(idx)
        
        # 최대 20개까지만 처리 (정확성 위해 적은 수)
        max_items = min(20, len(candidates))
        selected_candidates = candidates[:max_items]
        
        print(f"   📊 정확한 동의어 생성 대상: {len(candidates)}개 중 {max_items}개 처리")
        
        for idx in selected_candidates:
            row = fixed_df.loc[idx]
            word = str(row[word_col]).strip()
            meaning = str(row[meaning_col]).strip() if meaning_col else ""
            context = str(row[context_col]).strip() if context_col else ""
            
            # 정확한 동의어/반의어 생성
            result = self._generate_contextual_synonyms_antonyms(word, meaning, context)
            
            if result:
                synonyms = result.get('synonyms', '')
                antonyms = result.get('antonyms', '')
                
                if synonyms:
                    fixed_df.at[idx, synonyms_col] = synonyms
                    fix_count += 1
                    print(f"   ✅ '{word}': 동의어 '{synonyms}' 추가")
                
                if antonyms_col and antonyms:
                    fixed_df.at[idx, antonyms_col] = antonyms
                    print(f"   ✅ '{word}': 반의어 '{antonyms}' 추가")
        
        print(f"   📊 정확한 동의어/반의어 보완: {fix_count}개")
        return {"df": fixed_df, "fixed_count": fix_count}

    def _generate_contextual_synonyms_antonyms(self, word, meaning, context):
        """문맥과 의미를 고려한 정확한 동의어/반의어 생성"""
        # 캐시 확인
        cache_key = f"contextual_synonyms_{word}_{hash(meaning)}_{hash(context[:100])}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try: 
            prompt = f"""Find **exact synonyms and antonyms** that can be substituted for the given English word in this specific context.
**Word**: {word}
**Korean Meaning**: {meaning}
**Context**: {context[:300]}

Strict Requirements:
1. Synonyms: Words that can **exactly replace** '{word}' in this context with identical meaning
2. Antonyms: Words that can **perfectly substitute** '{word}' with opposite meaning in this context
3. **Identical part of speech** required (noun→noun, verb→verb, adjective→adjective)
4. **Exact same meaning** (not just similar)
5. **Grammatically perfect** in this specific context
6. If there's ANY doubt, leave empty
7. Maximum 2 words each (only absolutely certain ones)

Response format:
{{
    "synonyms": "word1, word2",
    "antonyms": "word1, word2",
    "explanation": "reasoning for selection"
}}

Examples:
- Context: "The task was very difficult to complete"
difficult (adjective) → synonyms: "hard, challenging" (adjectives only), antonyms: "easy" (adjective only)

- Context: "He decided to analyze the data carefully"  
analyze (verb) → synonyms: "examine" (verb only), antonyms: "" (not certain)

- If ANY uncertainty: synonyms: "", antonyms: ""
"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()
            
            try:
                # JSON 파싱 시도
                result = json.loads(result_text)
                
                # 결과 검증 및 정리
                synonyms = self._clean_synonym_list(result.get('synonyms', ''))
                antonyms = self._clean_synonym_list(result.get('antonyms', ''))
                
                final_result = {
                    'synonyms': synonyms,
                    'antonyms': antonyms,
                    'explanation': result.get('explanation', '')
                }
                
                # 캐시 저장
                self.cache[cache_key] = final_result
                self.ai_call_count += 1
                
                return final_result
                
            except json.JSONDecodeError:
                # JSON이 아닌 경우 텍스트에서 추출
                return self._parse_synonym_text_response(result_text, word)

        except Exception as e:
            print(f"⚠️ 문맥적 동의어/반의어 생성 실패 ({word}): {e}")
            return None

    def _clean_synonym_list(self, synonym_string):
        """동의어 리스트 정리"""
        if not synonym_string or synonym_string.strip() == "":
            return ""
        
        # 쉼표로 분리하고 정리
        words = [w.strip() for w in synonym_string.split(',')]
        
        # 빈 문자열이나 원래 단어는 제외
        cleaned = [w for w in words if w and len(w) > 1]
        
        # 최대 3개까지만
        cleaned = cleaned[:3]
        
        return ', '.join(cleaned) if cleaned else ""

    def _parse_synonym_text_response(self, text, original_word):
        """텍스트 응답에서 동의어/반의어 추출"""
        synonyms = ""
        antonyms = ""
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 'synonyms' in line.lower() and ':' in line:
                synonyms = line.split(':', 1)[1].strip().strip('"')
            elif 'antonyms' in line.lower() and ':' in line:
                antonyms = line.split(':', 1)[1].strip().strip('"')
        
        return {
            'synonyms': self._clean_synonym_list(synonyms),
            'antonyms': self._clean_synonym_list(antonyms),
            'explanation': 'Parsed from text response'
        }
    def print_detailed_quality_report(self):
        """상세 품질 보고서 출력 (한글 뜻 및 동의어 보완 포함)"""
        results = self.generate_contextual_quality_report()

        print(f"\n📊 종합 문맥 품질 보고서")
        print(f"=" * 70)
        print(f"📁 파일: {os.path.basename(self.vocabulary_file)}")
        print(f"📊 총 항목 수: {results['total_entries']}개")
        print(f"🎯 종합 품질 점수: {results['quality_score']:.1f}/100")
        print(f"🤖 AI 호출 횟수: {results['ai_calls_used']}회")

        # 품질 평가
        score = results["quality_score"]
        if score >= 90:
            print(f"   ✅ 우수한 품질! 교육용으로 완벽")
        elif score >= 75:
            print(f"   🟡 양호한 품질")
        elif score >= 60:
            print(f"   🟠 품질 개선 필요")
        else:
            print(f"   🔴 심각한 품질 문제")

        # 문제 유형별 요약
        breakdown = results["quality_breakdown"]
        print(f"\n🔍 발견된 문제:")
        print(f"   📝 지문 내 중복: {breakdown['within_passage_duplicates']}개")
        print(f"   🔗 숙어-단어 충돌: {breakdown['idiom_word_conflicts']}개")
        print(f"   🇰🇷 한글 뜻 누락/오류: {breakdown['missing_korean_meanings']}개")  # 🔥 새로 추가
        print(f"   🔗 동의어 누락: {breakdown['missing_synonyms']}개")  # 🔥 새로 추가
        print(f"   🎯 문맥-의미 부적합: {breakdown['context_meaning_mismatches']}개")
        print(f"   🔍 단어-의미 불일치: {breakdown['word_meaning_mismatches']}개")
        print(f"   📊 총 문제 수: {breakdown['total_issues']}개")

        # 주요 문제 상세 정보 (상위 5개)
        if results["issues_found"]:
            print(f"\n⚠️ 주요 문제 상세:")
            for i, issue in enumerate(results["issues_found"][:5], 1):
                print(f"\n{i}. {issue['description']}")
                if issue["type"] == "idiom_word_conflict":
                    print(f"   💡 권장사항: {issue.get('recommendation', '')}")
                elif issue["type"] == "missing_korean_meaning":
                    print(f"   💡 권장사항: {issue.get('recommendation', '')}")
                elif issue["type"] == "missing_synonyms":
                    print(f"   💡 권장사항: {issue.get('recommendation', '')}")
                elif "ai_analysis" in issue and issue["ai_analysis"]:
                    print(f"   🤖 AI 분석: {issue['ai_analysis']}")
                if "suggested_meaning" in issue and issue["suggested_meaning"]:
                    print(f"   💡 제안 의미: {issue['suggested_meaning']}")

            print(f"=" * 70)
        return results

    def export_quality_issues(self, output_file="quality_issues.xlsx"):
        """품질 문제를 Excel로 내보내기 (한글 뜻 및 동의어 보완 포함)"""
        results = self.generate_contextual_quality_report()

        if not results["issues_found"]:
            print("✅ 내보낼 품질 문제가 없습니다!")
            return

        # 문제들을 DataFrame으로 변환
        issues_data = []
        for issue in results["issues_found"]:
            issues_data.append(
                {
                    "문제유형": issue["type"],
                    "심각도": issue["severity"],
                    "지문ID": issue.get("passage_id", ""),
                    "단어": issue.get("word", ""),
                    "숙어": issue.get("idiom", ""),
                    "충돌단어": issue.get("conflicting_word", ""),
                    "현재의미": issue.get("current_meaning", ""),  # 🔥 새로 추가
                    "문제유형상세": issue.get("problem_type", ""),  # 🔥 새로 추가
                    "의미": issue.get("meaning", ""),
                    "문맥": issue.get("context", ""),
                    "문제설명": issue["description"],
                    "권장사항": issue.get("recommendation", ""),
                    "AI분석": issue.get("ai_analysis", ""),
                    "제안의미": issue.get("suggested_meaning", ""),
                    "행번호": issue.get("index", ""),
                }
            )

        issues_df = pd.DataFrame(issues_data)
        issues_df.to_excel(output_file, index=False)
        print(f"📊 품질 문제 리포트 저장: {output_file}")
        print(f"   📋 총 {len(issues_data)}개 문제 기록됨")

        # 문제 유형별 통계
        type_counts = issues_df["문제유형"].value_counts()
        print(f"   📈 문제 유형별 분포:")
        for issue_type, count in type_counts.items():
            type_name = {
                "within_passage_duplicate": "지문내중복",
                "idiom_word_conflict": "숙어-단어충돌",
                "missing_korean_meaning": "한글뜻누락/오류",  # 🔥 새로 추가
                "missing_synonyms": "동의어누락",  # 🔥 새로 추가
                "context_meaning_mismatch": "문맥-의미부적합",
                "word_meaning_mismatch": "단어-의미불일치",
            }.get(issue_type, issue_type)
            print(f"      • {type_name}: {count}개")

    # 🔥 새로운 기능: 한글 뜻 및 동의어 일괄 생성
    def enhance_vocabulary_meanings_and_synonyms(self, max_items=50):
        """한글 뜻 및 동의어 일괄 생성/보완"""
        print(f"🚀 한글 뜻 및 동의어 일괄 보완 시작 (최대 {max_items}개)...")
        
        if "meaning" not in self.actual_columns or "word" not in self.actual_columns:
            print("⚠️ 필수 컬럼을 찾을 수 없음")
            return self.df

        enhanced_df = self.df.copy()
        meaning_col = self.actual_columns["meaning"]
        word_col = self.actual_columns["word"]
        context_col = self.actual_columns.get("context", None)
        synonyms_col = self.actual_columns.get("synonyms", None)

        # 1. 한글 뜻 보완 대상 찾기
        problematic_meanings = enhanced_df[
            enhanced_df[meaning_col].astype(str).str.endswith("의 의미") |
            enhanced_df[meaning_col].astype(str).str.endswith("의미") |
            enhanced_df[meaning_col].isna() |
            (enhanced_df[meaning_col].astype(str).str.len() < 2)
        ].head(max_items // 2)

        print(f"📝 한글 뜻 보완 대상: {len(problematic_meanings)}개")

        for idx, row in problematic_meanings.iterrows():
            word = str(row[word_col]).strip()
            context = str(row[context_col]).strip() if context_col else ""
            
            new_meaning = self._generate_korean_meaning_with_ai(word, context)
            if new_meaning and new_meaning != f"{word}의 의미":
                enhanced_df.at[idx, meaning_col] = new_meaning
                print(f"   ✅ '{word}': 한글 뜻 생성 완료")

        # 2. 동의어 보완 (동의어 컬럼이 있는 경우만)
        if synonyms_col:
            empty_synonyms = enhanced_df[
                enhanced_df[synonyms_col].isna() |
                (enhanced_df[synonyms_col].astype(str).str.strip() == "") |
                (enhanced_df[synonyms_col].astype(str).str.strip() == "nan")
            ].head(max_items // 2)

            print(f"🔗 동의어 보완 대상: {len(empty_synonyms)}개")

            for idx, row in empty_synonyms.iterrows():
                word = str(row[word_col]).strip()
                context = str(row[context_col]).strip() if context_col else ""
                
                synonyms, antonyms = self._generate_synonyms_with_ai(word, context)
                if synonyms:
                    enhanced_df.at[idx, synonyms_col] = synonyms
                    print(f"   ✅ '{word}': 동의어 생성 완료")

        print(f"🎯 일괄 보완 완료! AI 호출: {self.ai_call_count}회")
        return enhanced_df