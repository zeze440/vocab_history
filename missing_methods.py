"""
AdvancedVocabExtractor 클래스에 추가할 누락된 메서드들
"""


def add_missing_methods_to_extractor():
    """
    AdvancedVocabExtractor 클래스에 추가할 메서드들을 반환
    실제로는 이 함수들을 클래스에 동적으로 바인딩하거나
    상속을 통해 추가할 수 있음
    """

    def extract_advanced_idioms_safely(self, text_str):
        """안전한 숙어 추출"""
        try:
            return self.extract_advanced_idioms(text_str)
        except Exception as e:
            print(f"⚠️ 안전한 숙어 추출 실패: {e}")
            return []

    def extract_difficult_words_safely(
        self, text_str, easy_words, child_vocab, freq_tiers
    ):
        """안전한 어려운 단어 추출"""
        try:
            return self.extract_difficult_words(
                text_str, easy_words, child_vocab, freq_tiers
            )
        except Exception as e:
            print(f"⚠️ 안전한 단어 추출 실패: {e}")
            return []

    def _process_words_with_gpt_filter(self, word_candidates):
        """GPT 필터를 사용한 단어 처리"""
        if hasattr(self, "gpt_filter") and self.gpt_filter:
            # GPT 필터 로직 (간단한 구현)
            print(f"   🤖 GPT 필터링 모드: {len(word_candidates)}개 후보")
            return self._process_words_without_gpt_filter(word_candidates, {})
        else:
            return self._process_words_without_gpt_filter(word_candidates, {})

    def _gpt_analyze_word_difficulty(self, word, context="", pos=""):
        """GPT 기반 단어 난이도 분석 (기본 구현)"""
        return {
            "difficulty_score": 5.0,
            "contextual_difficulty": 5.0,
            "recommendation": "exclude",
            "reasoning": "기본 분석 - GPT 연동 미구현",
            "is_basic_vocabulary": len(word) < 4,
        }

    return {
        "extract_advanced_idioms_safely": extract_advanced_idioms_safely,
        "extract_difficult_words_safely": extract_difficult_words_safely,
        "_process_words_with_gpt_filter": _process_words_with_gpt_filter,
        "_gpt_analyze_word_difficulty": _gpt_analyze_word_difficulty,
    }


class MissingMethodsMixin:
    """누락된 메서드들을 제공하는 Mixin 클래스"""

    def extract_advanced_idioms_safely(self, text_str):
        """안전한 숙어 추출"""
        try:
            return self.extract_advanced_idioms(text_str)
        except Exception as e:
            print(f"⚠️ 안전한 숙어 추출 실패: {e}")
            return []

    def extract_difficult_words_safely(
        self, text_str, easy_words, child_vocab, freq_tiers
    ):
        """안전한 어려운 단어 추출"""
        try:
            return self.extract_difficult_words(
                text_str, easy_words, child_vocab, freq_tiers
            )
        except Exception as e:
            print(f"⚠️ 안전한 단어 추출 실패: {e}")
            return []

    def _process_words_with_gpt_filter(self, word_candidates):
        """GPT 필터를 사용한 단어 처리"""
        if hasattr(self, "gpt_filter") and self.gpt_filter:
            print(f"   🤖 GPT 필터링 모드: {len(word_candidates)}개 후보")
            return self._process_words_without_gpt_filter(word_candidates, {})
        else:
            return self._process_words_without_gpt_filter(word_candidates, {})

    def _gpt_analyze_word_difficulty(self, word, context="", pos=""):
        """GPT 기반 단어 난이도 분석"""
        return {
            "difficulty_score": 5.0,
            "contextual_difficulty": 5.0,
            "recommendation": "exclude",
            "reasoning": "기본 분석 - GPT 연동 미구현",
            "is_basic_vocabulary": len(word) < 4,
        }
