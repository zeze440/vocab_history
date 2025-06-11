# extractor_methods.py - 추출 메서드들 (간단 수정 버전)

import re
from utils import force_extract_text, get_sentence_context, get_simple_pos


class ExtractorMethods:
    """추출기의 핵심 메서드들을 포함하는 믹스인 클래스"""

    def process_text_with_metadata(
        self, text_input, text_id, easy_words, child_vocab, freq_tiers
    ):
        """메타데이터를 포함한 텍스트 처리 - 새 메서드"""

        print(f"\n🔍 텍스트 {text_id} 처리 시작 (메타데이터 포함)...")

        # ✅ 메타데이터 추출
        passage_info = self.get_passage_info(text_input)

        # 텍스트 내용 추출
        if isinstance(text_input, dict):
            text_content = text_input.get("content", "")
        elif isinstance(text_input, str):
            text_content = text_input
        else:
            print(f"❌ 지원하지 않는 텍스트 형식: {type(text_input)}")
            return []

        # 🔥 입력값 검증
        if not text_content or text_content is None:
            print(f"❌ 텍스트 {text_id}: 빈 텍스트")
            return []

        try:
            text_str = force_extract_text(text_content)
            if not text_str or len(text_str.strip()) < 10:
                print(f"❌ 텍스트 {text_id}: 유효하지 않은 텍스트")
                return []
        except Exception as e:
            print(f"❌ 텍스트 {text_id}: 텍스트 추출 실패 - {e}")
            return []

        print(f"   📝 텍스트 길이: {len(text_str)}자")

        # ✅ 메타데이터 정보 출력
        if passage_info.get("textbook_studio_passage_title"):
            print(f"   📖 지문 제목: {passage_info['textbook_studio_passage_title']}")
        if passage_info.get("book_title"):
            print(f"   📚 교재명: {passage_info['book_title']}")

        rows = []
        stats = {
            "user_db_idioms": 0,
            "grammar_patterns": 0,
            "reference_idioms": 0,
            "user_db_words": 0,
            "gpt_words": 0,
        }

        try:
            # 1. 숙어 추출
            print(f"   🔍 숙어 추출 중...")
            try:
                idioms = self.extract_advanced_idioms(text_str)
                if idioms is None:
                    idioms = []

                print(f"   📊 숙어 추출 결과: {len(idioms)}개")

                for i, idiom in enumerate(idioms):
                    if idiom and isinstance(idiom, dict):
                        try:
                            # ✅ 메타데이터를 포함한 행 생성
                            row = self.create_result_row_with_metadata(
                                item=idiom,
                                passage_info=passage_info,
                                item_type="idiom",
                                order=i + 1,
                            )
                            rows.append(row)

                            # 통계 업데이트
                            if idiom.get("user_db_match"):
                                stats["user_db_idioms"] += 1
                            elif idiom.get("type") == "grammar_pattern":
                                stats["grammar_patterns"] += 1
                            else:
                                stats["reference_idioms"] += 1

                        except Exception as e:
                            print(f"         ❌ 숙어 {i+1} 처리 실패: {e}")

            except Exception as e:
                print(f"   ❌ 숙어 추출 실패: {e}")

            # 2. 어려운 단어 추출
            print(f"   🔍 어려운 단어 추출 중...")
            try:
                difficult_words = self.extract_difficult_words(
                    text_str, easy_words, child_vocab, freq_tiers
                )
                if difficult_words is None:
                    difficult_words = []

                print(f"   📊 어려운 단어 추출 결과: {len(difficult_words)}개")

                for i, word in enumerate(difficult_words):
                    if word and isinstance(word, dict):
                        try:
                            # ✅ 메타데이터를 포함한 행 생성
                            row = self.create_result_row_with_metadata(
                                item=word,
                                passage_info=passage_info,
                                item_type="word",
                                order=len(rows) + i + 1,  # 숙어 다음 순서
                            )
                            rows.append(row)

                            # 통계 업데이트
                            if word.get("user_db_match"):
                                stats["user_db_words"] += 1
                            else:
                                stats["gpt_words"] += 1

                        except Exception as e:
                            print(f"         ❌ 단어 {i+1} 처리 실패: {e}")

            except Exception as e:
                print(f"   ❌ 어려운 단어 추출 실패: {e}")

            # 통계 출력
            print(f"✅ 텍스트 {text_id} 처리 완료:")
            print(f"   📊 사용자 DB 숙어: {stats['user_db_idioms']}개")
            print(f"   📊 문법 패턴: {stats['grammar_patterns']}개")
            print(f"   📊 참조 DB 숙어: {stats['reference_idioms']}개")
            print(f"   📊 사용자 DB 단어: {stats['user_db_words']}개")
            print(f"   📊 GPT 선택 단어: {stats['gpt_words']}개")
            print(f"   📊 총 추출: {len(rows)}개")

            # 🔥 동의어/반의어 추가
            if rows:  # 결과가 있을 때만
                try:
                    print(f"   🔍 동의어/반의어 추가 중...")
                    rows = self.add_synonyms_antonyms_to_results(rows)
                    print(f"   ✅ 동의어/반의어 추가 완료: {len(rows)}개 항목")
                except Exception as e:
                    print(f"   ⚠️ 동의어/반의어 추가 실패: {e}")

        except Exception as e:
            print(f"❌ 텍스트 {text_id} 전체 처리 실패: {e}")
            import traceback

            traceback.print_exc()
            return []

        print(f"   🎯 최종 반환: {len(rows)}개 항목")
        return rows if rows else []

    def create_result_row_with_metadata(self, item, passage_info, item_type, order):
        """메타데이터를 포함한 결과 행 생성"""

        # ✅ 22컬럼 구조에 맞춰 생성
        row = {
            # 메타데이터 컬럼들
            "교재ID": passage_info.get("textbook_id", ""),
            "교재명": passage_info.get("book_title", "Advanced Vocabulary"),
            "지문ID": passage_info.get("textbook_studio_passage_id", ""),
            "순서": order,
            "지문": (
                passage_info.get("content", "")[:100] + "..."
                if len(passage_info.get("content", "")) > 100
                else passage_info.get("content", "")
            ),
            # 단어/숙어 정보
            "단어": item.get("original", ""),
            "원형": item.get("base_form", ""),
            "품사": item.get("pos", ""),
            "뜻(한글)": item.get("meaning", item.get("korean_meaning", "")),
            "뜻(영어)": "",  # 기본적으로 비움
            # 동의어/반의어 (나중에 추가됨)
            "동의어": "",
            "반의어": "",
            # 문맥 정보
            "문맥": item.get("context", ""),
            "분리형여부": item.get("is_separated", False),
            "신뢰도": item.get("confidence", 0.0),
            "사용자DB매칭": item.get("user_db_match", False),
            "매칭방식": item.get("match_type", "일반"),
            # 추가 정보
            "패턴정보": f"Studio: {passage_info.get('studio_title', '')}, Unit: {passage_info.get('textbook_unit_id', '')}",
            "문맥적의미": "",
            "동의어신뢰도": 0.0,
            "처리방식": "",
            "포함이유": f"지문 '{passage_info.get('textbook_studio_passage_title', '')}' 에서 추출",
        }

        return row

    def add_synonyms_antonyms_to_results(self, results):
        """동의어/반의어 추가 - 컬럼명 매핑 수정"""

        if not hasattr(self, "synonym_extractor") or not self.synonym_extractor:
            print("⚠️ 동의어/반의어 추출기가 없습니다")
            # 빈 컬럼 추가
            for result in results:
                result["동의어"] = ""
                result["반의어"] = ""
                result["동의어신뢰도"] = 0.0
            return results

        print("🔍 동의어/반의어 추출 중...")

        try:
            # 단어 정보 준비
            word_list = []
            for result in results:
                word_info = {
                    # ✅ 정확한 컬럼명 사용
                    "word": result.get("단어", ""),
                    "context": result.get("문맥", ""),
                    "pos": result.get("품사", ""),
                    "meaning": result.get("뜻(한글)", ""),
                }
                word_list.append(word_info)

            # 배치 추출
            synonym_results = self.synonym_extractor.batch_extract(word_list)

            # 결과에 추가
            for result in results:
                word = result.get("단어", "")
                if word in synonym_results:
                    syn_data = synonym_results[word]
                    synonyms = syn_data.get("synonyms", [])[:3]  # 최대 3개
                    antonyms = syn_data.get("antonyms", [])[:2]  # 최대 2개

                    # ✅ 정확한 컬럼명으로 저장
                    result["동의어"] = ", ".join(synonyms)
                    result["반의어"] = ", ".join(antonyms)
                    result["동의어신뢰도"] = syn_data.get("confidence", 0.0)
                else:
                    result["동의어"] = ""
                    result["반의어"] = ""
                    result["동의어신뢰도"] = 0.0

            return results

        except Exception as e:
            print(f"❌ 동의어/반의어 추출 실패: {e}")
            # 실패 시 빈 컬럼 추가
            for result in results:
                result["동의어"] = ""
                result["반의어"] = ""
                result["동의어신뢰도"] = 0.0
            return results

    def extract_advanced_idioms(self, text):
        """기본 숙어 추출"""
        results = []

        try:
            text_str = force_extract_text(text)
            if not text_str:
                return []

            found_positions = set()

            # 1. 사용자 DB 숙어 우선 검사
            print(f"      🔍 사용자 DB 숙어 검사...")
            try:
                if hasattr(self, "user_idioms") and self.user_idioms:
                    extracted_user_idioms = self.extract_user_db_idioms(text_str)
                    if extracted_user_idioms:
                        results.extend(extracted_user_idioms)
                        print(
                            f"      ✅ 사용자 DB 숙어: {len(extracted_user_idioms)}개"
                        )
                        # 위치 기록
                        for idiom in extracted_user_idioms:
                            if idiom and isinstance(idiom, dict):
                                base_form = idiom.get("base_form", "")
                                if base_form:
                                    start = text_str.lower().find(base_form.lower())
                                    if start != -1:
                                        end = start + len(base_form)
                                        found_positions.add((start, end))
                    else:
                        print(f"      ⚠️ 사용자 숙어 DB 없음")
            except Exception as e:
                print(f"      ❌ 사용자 DB 숙어 추출 실패: {e}")

            # 2. 참조 DB 숙어 검사 (간단 버전)
            print(f"      🔍 참조 DB 숙어 검사...")
            try:
                if hasattr(self, "reference_idioms") and self.reference_idioms:
                    ref_count = 0
                    for idiom in self.reference_idioms[:50]:  # 처음 50개만 확인
                        if idiom and idiom.lower() in text_str.lower():
                            start = text_str.lower().find(idiom.lower())
                            end = start + len(idiom)

                            # 위치 중복 확인
                            if any(abs(start - pos[0]) < 10 for pos in found_positions):
                                continue

                            context = get_sentence_context(text_str, start, end)
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
                            ref_count += 1

                    if ref_count > 0:
                        print(f"      ✅ 참조 DB 숙어: {ref_count}개")
                else:
                    print(f"      ⚠️ 참조 숙어 DB 없음")
            except Exception as e:
                print(f"      ❌ 참조 DB 숙어 추출 실패: {e}")

        except Exception as e:
            print(f"❌ 숙어 추출 전체 실패: {e}")
            return []

        return results

    def extract_user_db_idioms(self, text):
        """사용자 DB에서 숙어 추출"""
        results = []

        try:
            text_str = force_extract_text(text)
            text_lower = text_str.lower()
            found_positions = set()

            if not hasattr(self, "user_idioms") or not self.user_idioms:
                return results

            # 길이순 정렬
            sorted_user_idioms = sorted(self.user_idioms, key=len, reverse=True)

            for idiom in sorted_user_idioms:
                try:  # 개별 숙어 처리용 try-except
                    pattern = r"\b" + re.escape(idiom) + r"\b"
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)

                    for match in matches:
                        start, end = match.span()

                        # 위치 중복 확인
                        if any(abs(start - pos[0]) < 5 for pos in found_positions):
                            continue

                        context = get_sentence_context(text_str, start, end)
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
                                "match_type": "사용자DB숙어",
                            }
                        )
                        found_positions.add((start, end))

                except Exception as e:
                    print(f"         ❌ 숙어 '{idiom}' 처리 실패: {e}")
                    continue  # 이제 for 루프 안에 있으므로 정상 작동

        except Exception as e:
            print(f"❌ 사용자 DB 숙어 추출 전체 실패: {e}")

        return results

    def extract_difficult_words(self, text, easy_words, child_vocab, freq_tiers):
        """기본 어려운 단어 추출"""

        try:
            text_str = force_extract_text(text)
            word_candidates = []
            user_db_candidates = []
            seen_lemmas = set()

            # SpaCy로 토큰 분석
            if hasattr(self, "nlp") and self.nlp:
                doc = self.nlp(text_str)
                for token in doc:
                    word = token.text.lower()
                    lemma = token.lemma_.lower()
                    original_word = token.text

                    # 🔥 고유명사 제외 (새로 추가)
                    if token.pos_ in ["PROPN"] or token.ent_type_ in [
                        "PERSON",
                        "GPE",
                        "ORG",
                        "NORP",
                    ]:
                        if self.verbose:
                            print(
                                f"         ⚠️ 고유명사 제외: '{original_word}' ({token.pos_}, {token.ent_type_})"
                            )
                        continue

                    # 🔥 대문자로 시작하는 단어 제외 (추가 안전장치)
                    if original_word[0].isupper() and len(original_word) > 1:
                        if self.verbose:
                            print(
                                f"         ⚠️ 대문자 시작 단어 제외: '{original_word}'"
                            )
                        continue

                    # 기본 필터링
                    if (
                        len(word) < 3
                        or not word.isalpha()
                        or token.is_stop
                        or token.pos_ in ["PUNCT", "SPACE", "SYM"]
                    ):
                        continue
                    seen_lemmas.add(lemma)

                    # 문맥 추출
                    context = get_sentence_context(
                        text_str, token.idx, token.idx + len(token.text)
                    )

                    word_info = {
                        "word": original_word,
                        "lemma": lemma,
                        "context": context,
                        "pos": get_simple_pos(token.pos_),
                        "token_info": {
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "original": original_word,
                        },
                    }

                    # 사용자 DB 단어 우선 분리
                    if hasattr(self, "user_single_words") and (
                        lemma in self.user_single_words
                        or word in self.user_single_words
                    ):
                        user_db_candidates.append(word_info)
                        print(
                            f"         ✅ 사용자 DB 단어 발견: '{original_word}' (원형: {lemma})"
                        )
                    else:
                        # 간단 필터링
                        if lemma not in easy_words and (
                            not child_vocab or lemma not in child_vocab
                        ):
                            word_candidates.append(word_info)

            final_words = []

            # 사용자 DB 단어들 처리
            print(f"      👤 사용자 DB 단어 처리: {len(user_db_candidates)}개")
            for word_info in user_db_candidates:
                korean_meaning = self.enhanced_korean_definition(
                    word_info["lemma"], word_info["context"], is_phrase=False
                )

                word_result = {
                    "original": word_info["token_info"]["original"],
                    "base_form": word_info["lemma"],
                    "lemma": word_info["lemma"],
                    "pos": word_info["pos"],
                    "korean_meaning": korean_meaning,
                    "context": word_info["context"],
                    "difficulty_score": 8.0,
                    "difficulty_level": "user_priority",
                    "confidence": 1.0,
                    "inclusion_reason": "사용자DB우선포함",
                    "user_db_match": True,
                    "match_type": "사용자DB단어",
                }
                final_words.append(word_result)

            # 나머지 단어들 처리 (간단한 필터링)
            print(f"      🔍 일반 단어 처리: {len(word_candidates)}개")
            for word_info in word_candidates[:10]:  # 최대 10개만 처리
                lemma = word_info["lemma"]

                # 길이 기반 간단 필터링
                if len(lemma) >= 5:  # 5글자 이상만 포함
                    korean_meaning = self.enhanced_korean_definition(
                        lemma, word_info["context"], is_phrase=False
                    )

                    word_result = {
                        "original": word_info["token_info"]["original"],
                        "base_form": lemma,
                        "lemma": lemma,
                        "pos": word_info["pos"],
                        "korean_meaning": korean_meaning,
                        "context": word_info["context"],
                        "difficulty_score": 6.0,
                        "difficulty_level": "intermediate",
                        "confidence": 0.8,
                        "inclusion_reason": "길이기반선택",
                        "user_db_match": False,
                        "match_type": "일반단어",
                    }
                    final_words.append(word_result)

            user_db_count = len(
                [w for w in final_words if w.get("user_db_match", False)]
            )
            other_count = len(
                [w for w in final_words if not w.get("user_db_match", False)]
            )

            print(
                f"      📊 단어 추출 결과: 사용자DB {user_db_count}개 + 기타 {other_count}개 = 총 {len(final_words)}개"
            )

            return final_words

        except Exception as e:
            print(f"❌ 어려운 단어 추출 실패: {e}")
            return []

    def enhanced_korean_definition(
        self, word, sentence, is_phrase=False, pos_hint=None
    ):
        """한글 의미 생성 - 문맥 기반 개선"""

        try:
            # 직접 GPT 호출
            if not hasattr(self, "client") or not self.client:
                return f"{word}의 의미"

            # 🔥 문맥 기반 프롬프트 강화 (영어)
            if sentence and len(sentence.strip()) > 10:
                prompt = f"""Analyze the exact meaning of "{word}" in this sentence.

Context: "{sentence}"

Task: Determine what "{word}" means in THIS specific usage.

Instructions:
1. Read the sentence carefully
2. Understand how "{word}" functions in this context
3. Determine the specific meaning (not all possible meanings)
4. Provide ONLY the Korean equivalent for this specific usage

Response format: Just the Korean meaning (2-4 words)

Example:
- "bank" in "I went to the bank" → "은행"
- "bank" in "river bank" → "강둑"
- "light" in "light meal" → "가벼운"
- "different" in "He's a different kind of person" → "특별한"

Korean meaning of "{word}" in this context:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert English-Korean translator who analyzes context precisely. Provide the exact Korean meaning that fits the specific context, not just general dictionary definitions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=50,
            )

            answer = response.choices[0].message.content.strip().replace('"', "")

            # 🔥 추가 정제: 불필요한 설명 제거
            answer = answer.split("\n")[0]  # 첫 번째 줄만 사용
            answer = answer.replace("meaning:", "").replace("translation:", "").strip()

            # GPT 호출 카운트 업데이트
            if hasattr(self, "gpt_call_count"):
                self.gpt_call_count += 1

            # 토큰 사용량 업데이트
            if hasattr(self, "gpt_token_usage") and hasattr(response, "usage"):
                usage = response.usage
                self.gpt_token_usage["prompt_tokens"] += usage.prompt_tokens
                self.gpt_token_usage["completion_tokens"] += usage.completion_tokens
                self.gpt_token_usage["total_tokens"] += usage.total_tokens

            return answer

        except Exception as e:
            print(f"   ❌ 의미 생성 실패 ({word}): {e}")
            return word

    def add_synonyms_antonyms_to_results(self, results):
        """동의어/반의어 추가 (기본 버전)"""

        # 🔍 이 조건에서 막히는지 확인
        if not hasattr(self, "synonym_extractor") or not self.synonym_extractor:
            print("⚠️ 동의어/반의어 추출기가 없습니다")
            for result in results:
                result["동의어"] = ""
                result["반의어"] = ""
            return results

        print("🔍 동의어/반의어 추출 중...")

        try:
            # 단어 정보 준비
            word_list = []
            for result in results:
                word_info = {
                    "word": result.get("단어", ""),
                    "context": result.get("문맥", ""),
                    "pos": result.get("품사", ""),
                    "meaning": result.get("뜻(한글)", ""),
                }
                word_list.append(word_info)

            # 배치 추출
            synonym_results = self.synonym_extractor.batch_extract(word_list)

            # 결과에 추가
            for result in results:
                word = result.get("단어", "")
                if word in synonym_results:
                    syn_data = synonym_results[word]
                    synonyms = syn_data.get("synonyms", [])[:3]  # 최대 3개
                    antonyms = syn_data.get("antonyms", [])[:2]  # 최대 2개

                    result["동의어"] = ", ".join(synonyms)
                    result["반의어"] = ", ".join(antonyms)
                else:
                    result["동의어"] = ""
                    result["반의어"] = ""

            return results

        except Exception as e:
            print(f"❌ 동의어/반의어 추출 실패: {e}")
            # 실패 시 빈 컬럼 추가
            for result in results:
                result["동의어"] = ""
                result["반의어"] = ""
            return results
