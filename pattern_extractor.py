import pandas as pd
import re
import nltk
from nltk import ngrams, word_tokenize
from collections import Counter, defaultdict
import spacy
from typing import List, Dict, Tuple
import argparse
import json

# NLTK 다운로드
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# SpaCy 로드
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("❌ SpaCy 영어 모델이 없습니다. 다음 명령어로 설치하세요:")
    print("python -m spacy download en_core_web_sm")
    exit(1)


class PatternExtractor:
    """지문DB에서 교육용 패턴을 추출하는 클래스"""

    def __init__(self):
        # 교육적으로 중요한 시작 단어들 (패턴 앞부분에 오면 가점)
        self.educational_starters = {
            "according",
            "as",
            "in",
            "on",
            "with",
            "by",
            "due",
            "because",
            "since",
            "while",
            "during",
            "after",
            "before",
            "until",
            "unless",
            "although",
            "though",
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "nevertheless",
            "nonetheless",
            "consequently",
            "hence",
            "thus",
            "instead",
            "rather",
            "besides",
            "additionally",
            "similarly",
            "likewise",
        }

        # 전치사 리스트 (전치사구 패턴 인식용)
        self.prepositions = {
            "about",
            "above",
            "across",
            "after",
            "against",
            "along",
            "among",
            "around",
            "at",
            "before",
            "behind",
            "below",
            "beneath",
            "beside",
            "between",
            "beyond",
            "by",
            "down",
            "during",
            "except",
            "for",
            "from",
            "in",
            "inside",
            "into",
            "like",
            "near",
            "of",
            "off",
            "on",
            "outside",
            "over",
            "through",
            "throughout",
            "till",
            "to",
            "toward",
            "under",
            "until",
            "up",
            "upon",
            "with",
            "within",
            "without",
        }

    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not isinstance(text, str):
            return ""

        # 특수문자 정리 (단, 하이픈과 어포스트로피는 보존)
        text = re.sub(r"[^\w\s\'-]", " ", text)

        # 다중 공백을 단일 공백으로
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()

    def extract_ngrams(
        self, texts: List[str], n: int, min_freq: int = 2
    ) -> Dict[str, int]:
        """N-gram 추출"""
        ngram_counts = Counter()

        for text in texts:
            if not isinstance(text, str) or len(text.strip()) == 0:
                continue

            clean_text = self.clean_text(text)
            if len(clean_text) < 5:  # 너무 짧은 텍스트 제외
                continue

            # 토큰화
            try:
                tokens = word_tokenize(clean_text)
                tokens = [
                    token for token in tokens if len(token) > 1 and token.isalpha()
                ]

                if len(tokens) < n:
                    continue

                # N-gram 생성
                for ngram in ngrams(tokens, n):
                    ngram_str = " ".join(ngram)

                    # 기본 필터링
                    if self.is_valid_ngram(ngram, n):
                        ngram_counts[ngram_str] += 1

            except Exception as e:
                print(f"⚠️ 토큰화 오류: {e}")
                continue

        # 최소 빈도 필터링
        return {
            ngram: count for ngram, count in ngram_counts.items() if count >= min_freq
        }

    def is_valid_ngram(self, ngram: Tuple[str, ...], n: int) -> bool:
        """N-gram이 교육적으로 유효한지 판단"""
        ngram_list = list(ngram)

        # 2. 첫 번째나 마지막 단어가 관사인 경우 제외
        if ngram_list[0] in ["a", "an", "the"] or ngram_list[-1] in ["a", "an", "the"]:
            return False

        # 4. 숫자만 있는 경우 제외
        if all(word.isdigit() for word in ngram_list):
            return False

        return True

    def calculate_educational_score(self, ngram: str, frequency: int, n: int) -> float:
        """교육적 중요도 점수 계산"""
        score = frequency * 1.0  # 기본 빈도 점수

        words = ngram.split()

        # 2. 전치사구 패턴 보너스
        if any(word in self.prepositions for word in words):
            score += 1.5

        # 3. 길이별 가중치 (3-4gram이 가장 유용)
        length_weights = {2: 1.0, 3: 1.5, 4: 2.0, 5: 1.2}
        score *= length_weights.get(n, 1.0)

        # 4. 특정 패턴 보너스
        pattern_bonuses = {
            r"\bas\s+\w+\s+as\b": 3.0,  # as ... as
            r"\bin\s+order\s+to\b": 3.0,  # in order to
            r"\bas\s+a\s+result\b": 2.5,  # as a result
            r"\bin\s+spite\s+of\b": 2.5,  # in spite of
            r"\bdue\s+to\b": 2.0,  # due to
            r"\baccording\s+to\b": 2.0,  # according to
            r"\bnot\s+only\b": 2.5,  # not only
            r"\bwould\s+rather\b": 2.0,  # would rather
        }

        for pattern, bonus in pattern_bonuses.items():
            if re.search(pattern, ngram, re.IGNORECASE):
                score += bonus
                break

        return score

    def filter_by_pos_patterns(self, ngrams_dict: Dict[str, int]) -> Dict[str, Dict]:
        """품사 패턴을 고려한 필터링"""
        filtered_results = {}

        for ngram, frequency in ngrams_dict.items():
            try:
                # SpaCy로 품사 분석
                doc = nlp(ngram)
                pos_pattern = [token.pos_ for token in doc]

                # 교육적으로 중요한 품사 패턴들
                important_patterns = [
                    ["ADP", "NOUN"],  # 전치사 + 명사
                    ["ADV", "ADJ"],  # 부사 + 형용사
                    ["VERB", "ADP"],  # 동사 + 전치사 (phrasal verb)
                    ["ADP", "NOUN", "ADP"],  # 전치사 + 명사 + 전치사
                    ["CONJ", "ADP"],  # 접속사 + 전치사
                    ["ADV", "CONJ"],  # 부사 + 접속사
                ]

                # 패턴 매칭 확인
                is_important = any(
                    pos_pattern == pattern
                    or (
                        len(pos_pattern) >= len(pattern)
                        and pos_pattern[: len(pattern)] == pattern
                    )
                    for pattern in important_patterns
                )

                if is_important or frequency >= 5:  # 고빈도는 품사 상관없이 포함
                    filtered_results[ngram] = {
                        "frequency": frequency,
                        "pos_pattern": pos_pattern,
                        "is_important_pattern": is_important,
                    }

            except Exception as e:
                # SpaCy 처리 실패시 빈도만으로 판단
                if frequency >= 3:
                    filtered_results[ngram] = {
                        "frequency": frequency,
                        "pos_pattern": [],
                        "is_important_pattern": False,
                    }

        return filtered_results

    def extract_comprehensive_patterns(self, texts: List[str]) -> Dict[int, Dict]:
        """종합적인 패턴 추출"""
        print("📊 N-gram 패턴 추출 시작...")

        all_patterns = {}

        # 2-gram부터 5-gram까지 추출
        for n in range(2, 6):
            print(f"🔍 {n}-gram 추출 중...")

            min_freq = max(1, 5 - n)  # 길수록 낮은 최소빈도
            ngrams_dict = self.extract_ngrams(texts, n, min_freq)

            print(f"   → {len(ngrams_dict)}개 {n}-gram 발견")

            # 품사 패턴 필터링
            filtered_ngrams = self.filter_by_pos_patterns(ngrams_dict)

            # 교육적 점수 계산
            scored_patterns = {}
            for ngram, info in filtered_ngrams.items():
                score = self.calculate_educational_score(ngram, info["frequency"], n)
                scored_patterns[ngram] = {
                    **info,
                    "educational_score": score,
                    "length": n,
                }

            # 상위 패턴만 선별 (n-gram별로 최대 50개)
            top_patterns = dict(
                sorted(
                    scored_patterns.items(),
                    key=lambda x: x[1]["educational_score"],
                    reverse=True,
                )[:200]
            )

            all_patterns[n] = top_patterns
            print(f"   ✅ {len(top_patterns)}개 상위 {n}-gram 선별")

        return all_patterns

    def export_patterns(self, patterns: Dict[int, Dict], output_file: str):
        """패턴을 Excel 파일로 저장"""
        all_data = []

        for n, ngrams in patterns.items():
            for ngram, info in ngrams.items():
                all_data.append(
                    {
                        "pattern": ngram,
                        "length": n,
                        "frequency": info["frequency"],
                        "educational_score": round(info["educational_score"], 2),
                        "pos_pattern": " -> ".join(info.get("pos_pattern", [])),
                        "is_important": info.get("is_important_pattern", False),
                        "category": self.categorize_pattern(ngram),
                    }
                )

        # DataFrame 생성 및 정렬
        df = pd.DataFrame(all_data)
        df = df.sort_values(
            ["educational_score", "frequency"], ascending=[False, False]
        )

        # Excel 저장
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # 전체 패턴
            df.to_excel(writer, sheet_name="전체패턴", index=False)

            # 길이별 시트
            for n in range(2, 6):
                n_gram_df = df[df["length"] == n]
                if not n_gram_df.empty:
                    n_gram_df.to_excel(writer, sheet_name=f"{n}gram", index=False)

            # 카테고리별 시트
            categories = df["category"].unique()
            for category in categories:
                cat_df = df[df["category"] == category]
                if not cat_df.empty and len(category) < 30:  # 시트명 길이 제한
                    cat_df.to_excel(writer, sheet_name=category, index=False)

        print(f"✅ 패턴 분석 결과가 {output_file}에 저장되었습니다.")
        print(f"📊 총 {len(df)}개 패턴 추출됨")

    def categorize_pattern(self, pattern: str) -> str:
        """패턴을 카테고리로 분류"""
        pattern_lower = pattern.lower()

        # 연결어구
        if any(
            word in pattern_lower
            for word in [
                "as a result",
                "in addition",
                "however",
                "therefore",
                "moreover",
                "furthermore",
            ]
        ):
            return "연결어구"

        # 전치사구
        elif any(
            pattern_lower.startswith(prep)
            for prep in ["in", "on", "at", "by", "with", "for", "of"]
        ):
            return "전치사구"

        # 시간표현
        elif any(
            word in pattern_lower
            for word in ["when", "while", "during", "before", "after", "until", "since"]
        ):
            return "시간표현"

        # 조건표현
        elif any(
            word in pattern_lower
            for word in ["if", "unless", "provided", "suppose", "in case"]
        ):
            return "조건표현"

        # 동사구
        elif len(pattern.split()) == 2 and any(
            word in pattern_lower for word in ["up", "out", "off", "on", "down", "away"]
        ):
            return "동사구"

        # 학술표현
        elif any(
            word in pattern_lower
            for word in ["according to", "due to", "because of", "in terms of"]
        ):
            return "학술표현"

        else:
            return "기타표현"


def main():
    parser = argparse.ArgumentParser(description="지문DB에서 교육용 패턴 추출")
    parser.add_argument("--input", "-i", required=True, help="입력 CSV 파일 (지문DB)")
    parser.add_argument(
        "--output", "-o", default="extracted_patterns.xlsx", help="출력 Excel 파일"
    )
    parser.add_argument(
        "--text-column", "-t", default="content", help="텍스트가 들어있는 컬럼명"
    )
    parser.add_argument("--encoding", "-e", default="cp949", help="CSV 파일 인코딩")

    args = parser.parse_args()

    try:
        # CSV 파일 읽기
        print(f"📂 {args.input} 파일 읽는 중...")
        try:
            df = pd.read_csv(args.input, encoding=args.encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(args.input, encoding="utf-8")

        print(f"✅ {len(df)}개 행 로드 완료")
        print(f"📊 컬럼명: {list(df.columns)}")

        # 텍스트 컬럼 찾기
        text_column = args.text_column
        if text_column not in df.columns:
            # 첫 번째 컬럼 사용
            text_column = df.columns[0]
            print(f"⚠️ '{args.text_column}' 컬럼이 없어서 '{text_column}' 컬럼 사용")

        # 텍스트 추출
        texts = df[text_column].dropna().astype(str).tolist()
        print(f"📝 {len(texts)}개 텍스트 추출 완료")

        # 패턴 추출기 초기화
        extractor = PatternExtractor()

        # 패턴 추출 실행
        patterns = extractor.extract_comprehensive_patterns(texts)

        # 결과 저장
        extractor.export_patterns(patterns, args.output)

        # 요약 통계 출력
        total_patterns = sum(len(ngrams) for ngrams in patterns.values())
        print(f"\n📈 추출 결과 요약:")
        for n, ngrams in patterns.items():
            print(f"  • {n}-gram: {len(ngrams)}개")
        print(f"  • 총 패턴: {total_patterns}개")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
