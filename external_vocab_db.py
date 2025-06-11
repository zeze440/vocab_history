# external_vocab_db.py - 외부 어휘 데이터베이스

import os
from config import BASIC_VERBS, BASIC_ADJECTIVES, BASIC_NOUNS


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
