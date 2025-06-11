# main_extractor.py - 메인 어휘 추출기

import os
import time
import pandas as pd
import quality_checker
from tqdm import tqdm
from config import DEFAULT_SETTINGS, FilePaths
from utils import (
    force_extract_text,
    get_sentence_context,
    get_simple_pos,
    clean_korean_definition,
    safe_read_csv,
    find_text_column,
)
from package_manager import get_nlp_model, get_openai_client, initialize_packages
from cache_manager import get_cache_manager
from difficulty_analyzer import GPTDifficultyFilter
from extractor_methods import ExtractorMethods
from separable_idiom_detector import SeparableIdiomDetector, AdvancedIdiomChecker
from external_vocab_db import ExternalVocabDatabase

# 필수 import들
try:
    from safe_data_utils import (
        safe_get_column_value,
        safe_string_operation,
        safe_numeric_operation,
    )
except ImportError:
    print("⚠️ safe_data_utils를 찾을 수 없습니다. 기본 구현을 사용합니다.")

    def safe_get_column_value(df, col, default=""):
        return df.get(col, default)

    def safe_string_operation(text, operation="strip"):
        return str(text).strip() if text else ""

    def safe_numeric_operation(value, operation="float"):
        try:
            return float(value) if value else 0.0
        except:
            return 0.0


try:
    from missing_methods import MissingMethodsMixin
except ImportError:
    print("⚠️ missing_methods를 찾을 수 없습니다. 빈 클래스를 사용합니다.")

    class MissingMethodsMixin:
        pass


class AdvancedVocabExtractor(ExtractorMethods, MissingMethodsMixin):
    """고급 패턴 분석 어휘 추출기"""

    def __init__(
        self,
        user_words_file=None,
        settings=None,
        csv_file=None,
        verbose=False,
        **kwargs,
    ):
        # 패키지 초기화
        initialize_packages()

        # 기본 설정
        self.settings = DEFAULT_SETTINGS.copy()
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

        # 디렉토리 생성
        FilePaths.ensure_directories()

        # 사용자 단어 DB 초기화
        self.user_words = set()
        self.user_idioms = set()  # 숙어만 따로 관리
        self.user_single_words = set()  # 단일 단어만 따로 관리

        # 기본 컴포넌트들
        self.nlp = get_nlp_model()
        self.client = get_openai_client()
        self.phrase_db = None

        # 캐시 관리자
        self.cache_manager = get_cache_manager()
        self.load_cache_from_file(self.settings["CACHE_FILE"])

        # 쉬운 단어 목록
        self.easy_words = self._load_easy_words()

        # 빈도 데이터
        self.freq_tiers = {}
        if csv_file and os.path.exists(csv_file):
            print(f"📊 '{csv_file}'에서 빈도 데이터 구축 중...")
            self.freq_tiers = self._build_frequency_from_csv(csv_file)

        # 동의어/반의어 추출기 초기화
        self._initialize_synonym_extractor()

        # 🔥 지문 DB 로딩 추가 (data_loader 활용)
        self.passage_db = None
        self.load_texts_data = None

        # 🔍 초기화 결과 확인 코드 추가
        print(
            f"🔍 최종 synonym_extractor 상태: {getattr(self, 'synonym_extractor', 'NOT_SET')}"
        )
        if hasattr(self, "synonym_extractor") and self.synonym_extractor:
            print("✅ 동의어/반의어 기능 활성화됨")
        else:
            print("❌ 동의어/반의어 기능 비활성화됨")

        # 고급 검증기 초기화
        self.idiom_checker = AdvancedIdiomChecker(self.nlp)

        # 외부 DB 초기화
        self.external_vocab_db = ExternalVocabDatabase()

        # 사용자 단어 파일 로딩
        user_words_file = user_words_file or FilePaths.USER_WORDS_FILE
        if user_words_file and os.path.exists(user_words_file):
            print(f"📖 사용자 단어 파일 로딩: {user_words_file}")
            self._load_user_words_with_idiom_detection(user_words_file)
        else:
            print(f"🔍 사용자 단어 파일을 찾을 수 없습니다: {user_words_file}")

        if csv_file and os.path.exists(csv_file):
            self.passage_db = data_loader.load_texts_data(csv_file)
        # GPT 기반 난이도 필터 초기화
        self._initialize_gpt_difficulty_filter()
        # 분리형 숙어 감지기 초기화
        self._initialize_separable_detection()

        # 참조 숙어 DB 로딩
        self._load_reference_idioms()

        self._print_initialization_summary()

    def _initialize_synonym_extractor(self):
        try:
            from synonym_antonym_module import SynonymAntonymExtractor

            self.synonym_extractor = SynonymAntonymExtractor(
                client=self.client,
                cache_file="synonym_antonym_cache.json",
                verbose=self.verbose,
            )
            print("✅ 동의어/반의어 추출기 초기화 완료")
        except ImportError as e:
            print(f"⚠️ synonym_antonym_module.py를 찾을 수 없습니다: {e}")
            self.synonym_extractor = None

        except ImportError as e:
            print(f"⚠️ synonym_antonym_module.py를 찾을 수 없습니다: {e}")
            print("   동의어/반의어 기능을 비활성화하고 계속 진행합니다")
            self.synonym_extractor = None

 
        # 초기화 결과 확인
        if self.synonym_extractor:
            print("📚 동의어/반의어 추출 기능 활성화됨")
        else:
            print("⚠️ 동의어/반의어 추출 기능 비활성화됨 (다른 기능은 정상 작동)")

    def _initialize_gpt_difficulty_filter(self):
        """GPT 기반 난이도 필터 초기화"""
        if not self.settings.get("USE_GPT_DIFFICULTY_FILTER", True):
            print("⚠️ GPT 난이도 필터가 설정에서 비활성화됨")
            return

        try:
            print("🤖 GPT 기반 난이도 필터 초기화 중...")
            print(f"   📊 사용자 단일 단어: {len(self.user_single_words)}개")

            self.gpt_filter = GPTDifficultyFilter(
                client=self.client,
                user_words=self.user_single_words,
                cache_file="gpt_difficulty_filter_cache.json",
            )

            print("✅ GPT 난이도 필터 초기화 완료")

        except Exception as e:
            print(f"⚠️ GPT 난이도 필터 초기화 실패: {e}")
            print("기존 방식으로 계속 진행합니다.")
            self.gpt_filter = None

    def _initialize_separable_detection(self):
        """분리형 숙어 감지 시스템 초기화"""
        print(f"🔧 분리형 숙어 감지 시스템 초기화...")

        # SeparableIdiomDetector 인스턴스 생성
        self.separable_detector = SeparableIdiomDetector(
            self.client, verbose=self.verbose
        )

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

    def _load_user_words_with_idiom_detection(self, user_words_file):
        """사용자 단어 파일에서 숙어와 단일 단어를 구분하여 로딩"""
        try:
            if user_words_file.endswith(".csv"):
                user_df = safe_read_csv(user_words_file)
                if user_df is None:
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

                # 숙어와 단일 단어 분리
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

                # 사용자 숙어 샘플 출력
                if self.user_idioms:
                    sample_idioms = list(self.user_idioms)[:5]
                    print(f"   📝 사용자 숙어 예시: {sample_idioms}")
            else:
                print(f"   ⚠️ 파일이 비어있거나 읽을 수 없음")

        except Exception as e:
            print(f"   ❌ 사용자 단어 파일 로드 실패: {e}")

    def _load_reference_idioms(self):
        """참조 숙어 DB 로딩"""
        from data_loader import load_custom_idioms_from_data_directory

        data_dir = self.settings.get("data_dir", FilePaths.DATA_DIR)
        self.reference_idioms = load_custom_idioms_from_data_directory(data_dir)

    def _load_easy_words(self):
        """쉬운 단어 목록 로드"""
        import pickle
        from nltk.corpus import stopwords
        from config import BASIC_VERBS, BASIC_ADJECTIVES, BASIC_NOUNS, EASY_WORDS_FILES

        try:
            # 1. 먼저 pickle 캐시 확인
            easy_words_cache = self.settings["EASY_WORDS_CACHE"]
            if os.path.exists(easy_words_cache):
                with open(easy_words_cache, "rb") as f:
                    easy_words = pickle.load(f)
                print(f"✅ 쉬운 단어 목록 {len(easy_words)}개 캐시에서 로드 완료")
                return easy_words

            # 2. Excel 파일 확인
            for excel_file in EASY_WORDS_FILES["excel"]:
                if os.path.exists(excel_file):
                    print(f"📊 Excel 파일에서 쉬운 단어 로딩: {excel_file}")
                    try:
                        df = pd.read_excel(excel_file)
                        words_column = df.columns[0]
                        easy_words = set(
                            df[words_column]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )

                        # pickle 캐시로 저장
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
            for csv_file in EASY_WORDS_FILES["csv"]:
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
        """CSV 파일에서 빈도 데이터 구축"""
        from collections import Counter
        import os

        try:
            print(f"📊 빈도 분석 시작: {csv_file}")

            df = safe_read_csv(csv_file)
            if df is None:
                return {}

            print(f"   📋 데이터프레임 크기: {len(df)} rows, {len(df.columns)} columns")
            print(f"   📋 컬럼명: {list(df.columns)}")

            # 텍스트 컬럼 찾기
            found_column = find_text_column(df)
            if not found_column:
                print(f"   ⚠️ 텍스트 컬럼을 찾을 수 없음")
                return {}

            print(f"   📝 사용할 컬럼: {found_column}")

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
        from collections import Counter

        word_counts = Counter()
        for text in texts:
            doc = self.nlp(text)
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

    def _print_initialization_summary(self):
        """초기화 완료 요약 출력"""
        print(f"✅ 고급 패턴 분석 추출기 초기화 완료")
        print(
            f"   • 통합 컨텍스트 의미 생성: {'✅' if self.settings['USE_INTEGRATED_CONTEXTUAL'] else '❌'}"
        )
        print(
            f"   • 통합 난이도 분석: {'✅' if self.settings['USE_INTEGRATED_DIFFICULTY'] else '❌'}"
        )
        print(f"   • 고급 구동사 분석: ✅")
        print(f"   • 문법 패턴 분석: ✅")

        print(f"\n📊 데이터 소스 로딩 상황:")
        print(f"=" * 60)
        print(f"   👤 사용자 전체 단어: {len(self.user_words)}개")
        print(f"   📝 사용자 숙어: {len(self.user_idioms)}개")
        print(f"   🔤 사용자 단일 단어: {len(self.user_single_words)}개")
        print(f"   🏛️ 참조 숙어 DB: {len(self.reference_idioms)}개")
        print(f"   📚 쉬운 단어: {len(self.easy_words)}개")
        print(f"   📊 빈도 데이터: {len(self.freq_tiers)}개")
        print(f"=" * 60)

    def load_cache_from_file(self, cache_file=None):
        """GPT 캐시 로드"""
        import json

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

    def save_cache_to_file(self, cache_file=None):
        """GPT 캐시 저장"""
        import json

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

    def get_passage_info(self, text_input):
        """텍스트 입력에서 지문 메타데이터 추출"""
        try:
            # text_input이 딕셔너리 형태인지 확인 (data_loader에서 온 경우)
            if isinstance(text_input, dict):
                return {
                    "textbook_id": text_input.get("textbook_id"),
                    "book_title": text_input.get("book_title", "Advanced Vocabulary"),
                    "textbook_studio_passage_id": text_input.get(
                        "textbook_studio_passage_id"
                    ),
                    "textbook_unit_id": text_input.get("textbook_unit_id"),
                    "studio_title": text_input.get("studio_title"),
                    "studio_series": text_input.get("studio_series"),
                    "studio_title2": text_input.get("studio_title2"),
                    "textbook_studio_passage_title": text_input.get(
                        "textbook_studio_passage_title"
                    ),
                    "passage_order": text_input.get("passage_order"),
                    "content": text_input.get("content", ""),
                }

            # 기본값 반환 (텍스트만 있는 경우)
            return {
                "textbook_id": None,
                "book_title": "Advanced Vocabulary",
                "textbook_studio_passage_id": None,
                "textbook_unit_id": None,
                "studio_title": "",
                "studio_series": "",
                "studio_title2": "",
                "textbook_studio_passage_title": "",
                "passage_order": 1,
                "content": str(text_input) if isinstance(text_input, str) else "",
            }

        except Exception as e:
            print(f"⚠️ 지문 정보 추출 실패: {e}")
            return {
                "textbook_id": None,
                "book_title": "Advanced Vocabulary",
                "textbook_studio_passage_id": None,
                "textbook_unit_id": None,
                "studio_title": "",
                "studio_series": "",
                "studio_title2": "",
                "textbook_studio_passage_title": "",
                "passage_order": 1,
                "content": "",
            }

    def generate_vocabulary_workbook(
        self, texts, output_file="vocabulary_advanced.xlsx", **kwargs
    ):
        """고급 패턴 분석 단어장 생성 - 수정된 버전"""
        start_time = time.time()

        print(f"🚀 고급 패턴 분석 단어장 생성 시작")
        print(f"   • 메타데이터 포함 처리: ✅")
        print(f"   • 사용자 DB 숙어: ✅ {len(self.user_idioms)}개 활용")
        print(f"   • 사용자 DB 단어: ✅ {len(self.user_single_words)}개 활용")

        # 진행률 표시와 함께 텍스트 처리
        results = []
        for idx, text_input in enumerate(
            tqdm(texts, desc="📝 텍스트 처리 중 (메타데이터 포함)", unit="지문")
        ):
            try:
                # ✅ 텍스트 입력 타입에 따라 처리
                if isinstance(text_input, dict):
                    # data_loader에서 온 메타데이터 포함 텍스트
                    text_content = text_input.get("content", "")
                    text_id = text_input.get("id", f"text_{idx + 1}")
                elif isinstance(text_input, str):
                    # 단순 텍스트
                    text_content = text_input
                    text_id = f"text_{idx + 1}"
                else:
                    print(f"⚠️ 지원하지 않는 텍스트 형식: {type(text_input)}")
                    continue

                # ✅ process_text_with_metadata 호출 (새 메서드)
                result = self.process_text_with_metadata(
                    text_input,  # 전체 입력 (메타데이터 포함)
                    text_id,
                    self.easy_words,
                    set(),
                    self.freq_tiers,
                )

                if result is not None and isinstance(result, (list, tuple)):
                    results.extend(result)
                else:
                    print(f"⚠️ 텍스트 {idx + 1}: 유효하지 않은 결과")

            except Exception as e:
                print(f"❌ 텍스트 {idx + 1} 처리 실패: {e}")
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

        # 패턴별 통계 출력
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

            checker = quality_checker.ContextualVocabularyQualityChecker(
                temp_file, self.client
            )
            results = checker.print_detailed_quality_report()  # ✅ 올바른 메서드

            print("🔧 품질 문제 자동 수정 중...")
            fixed_df = checker.fix_quality_issues(apply_fixes=True)  # ✅ 올바른 메서드
            if os.path.exists(temp_file):
                os.remove(temp_file)

            print(f"📊 품질 점수: {results['quality_score']:.1f}/100")
            print(f"📊 발견된 문제: {results['quality_breakdown']['total_issues']}개")

            return fixed_df, results

        except Exception as e:
            print(f"❌ 품질 검사 실패: {e}")
            return df, None

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

    def enhanced_extract_user_db_idioms_with_separable(self, text):
        """분리형 감지 기능이 추가된 사용자 DB 숙어 추출"""
        results = []
        text_str = force_extract_text(text)
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
                context = get_sentence_context(
                    text_str, sep_result["start"], sep_result["end"]
                )
                meaning = self.enhanced_korean_definition(
                    sep_result["display_form"], context, is_phrase=True
                )

                results.append(
                    {
                        "original": sep_result["original"],
                        "base_form": sep_result["display_form"],  # 분리형 표시 포함
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
            import re

            pattern = r"\b" + re.escape(idiom) + r"\b"
            matches = re.finditer(pattern, text_str, re.IGNORECASE)

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

    def is_word_appropriate_for_extraction(self, word, context="", pos=""):
        """통합된 단어 적합성 판별"""
        word_lower = word.lower()

        # 사용자 DB에 이미 있는 단어는 무조건 적합
        if word_lower in self.user_words:
            return True, "사용자DB포함"

        # 기본 단어 강력 차단
        from config import BASIC_VERBS, BASIC_ADJECTIVES, BASIC_NOUNS

        if (
            word_lower in BASIC_VERBS
            or word_lower in BASIC_ADJECTIVES
            or word_lower in BASIC_NOUNS
        ):
            return False, "기본단어제외"

        # 외부 DB 기본 어휘 차단
        from external_vocab_db import is_basic_by_external_db

        if is_basic_by_external_db(word_lower):
            return False, "외부DB기본어휘"

        # 사용자 DB 우선 포함
        if word_lower in self.user_single_words:
            return True, "사용자DB우선포함"

        # GPT 문맥별 난이도 분석
        if hasattr(self, "gpt_filter") and self.gpt_filter:
            appropriate, reason = self.gpt_filter.is_word_appropriate_for_user_db(
                word, context, pos
            )
            return appropriate, reason

        # 기본 판별 로직
        if len(word) < 4:
            return False, "길이부족"

        return True, "기본통과"
# main_extractor.py 파일 끝부분에 다음 메서드들을 추가하세요:

    def enhanced_korean_definition(self, word, context, is_phrase=False):
        """AI를 사용한 한글 뜻 생성"""
        # 캐시 확인
        cache_key = f"korean_def_{word}_{hash(context[:100])}_{is_phrase}"
        if cache_key in self.gpt_cache:
            return self.gpt_cache[cache_key]

        # GPT 호출 제한 확인
        if self.gpt_call_count >= self.GPT_CALL_LIMIT:
            return f"{word}의 의미"

        try:
            if is_phrase:
                prompt = f"""Please provide a simple and clear Korean meaning for the following English idiom/phrase.

**Idiom/Phrase**: {word}
**Context**: {context[:200]}

Requirements:
1. Answer in Korean only (no English words)
2. Keep it simple and clear (within 5 words)
3. Match the contextual meaning

Korean meaning:"""
            else:
                prompt = f"""Please provide a simple and clear Korean meaning for the following English word.

**Word**: {word}
**Context**: {context[:200]}

Requirements:
1. Answer in Korean only (no English words)
2. Keep it simple and clear (within 3 words)
3. Match the contextual meaning

Korean meaning:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip()
            
            # 결과 정리
            result = result.replace("한국어 뜻:", "").strip()
            result = result.strip('"\'')
            
            # 영어가 포함되거나 너무 길면 기본값
            if any(char.isalpha() and ord(char) < 128 for char in result) or len(result) > 20:
                result = f"{word}의 의미"
            
            # 캐시 저장
            self.gpt_cache[cache_key] = result
            self.gpt_call_count += 1
            
            return result

        except Exception as e:
            print(f"⚠️ 한글 뜻 생성 실패 ({word}): {e}")
            return f"{word}의 의미"

    def generate_synonyms_and_antonyms(self, word, context=""):
        """동의어와 반의어 생성"""
        # 내장 AI 기반 동의어/반의어 생성만 사용
        cache_key = f"synonyms_{word}_{hash(context[:100])}"
        if cache_key in self.gpt_cache:
            cached = self.gpt_cache[cache_key]
            return cached.get('synonyms', ''), cached.get('antonyms', '')

        if self.gpt_call_count >= self.GPT_CALL_LIMIT:
            return "", ""

        try:
            prompt = f"""Find synonyms and antonyms for the following English word.

**Word**: {word}
**Context**: {context[:200]}

Requirements:
1. Up to 3 synonyms (separated by commas)
2. Up to 2 antonyms (separated by commas)
3. Only words that fit the context
4. Empty string if none available

Format:
Synonyms: word1, word2, word3
Antonyms: word1, word2"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip()
            
            # 결과 파싱
            synonyms = ""
            antonyms = ""
            
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Synonyms:'):
                    synonyms = line.split(':', 1)[1].strip().replace(' ', '')
                elif line.startswith('Antonyms:'):
                    antonyms = line.split(':', 1)[1].strip().replace(' ', '')
            
            # 캐시 저장
            self.gpt_cache[cache_key] = {'synonyms': synonyms, 'antonyms': antonyms}
            self.gpt_call_count += 1
            
            return synonyms, antonyms

        except Exception as e:
            print(f"⚠️ 동의어/반의어 생성 실패 ({word}): {e}")
            return "", ""
    def process_text_with_metadata(self, text_input, text_id, easy_words, excluded_words, freq_tiers):
        """메타데이터를 포함한 텍스트 처리 - 한글 뜻 및 동의어 생성 포함"""
        results = []
        
        try:
            # 지문 정보 추출
            passage_info = self.get_passage_info(text_input)
            text_content = passage_info["content"]
            
            if not text_content.strip():
                return results

            # 1. 사용자 DB 숙어 추출 (기존 메서드 활용)
            user_idiom_results = self.enhanced_extract_user_db_idioms_with_separable(text_content)
            
            for result in user_idiom_results:
                # 동의어 생성
                synonyms, antonyms = self.generate_synonyms_and_antonyms(
                    result['base_form'], 
                    result['context']
                )
                
                # 결과에 추가 정보 포함
                vocab_entry = {
                    "교재ID": passage_info.get("textbook_id"),
                    "교재명": passage_info.get("book_title", "Advanced Vocabulary"),
                    "지문ID": passage_info.get("textbook_studio_passage_id", text_id),
                    "순서": len(results) + 1,
                    "지문": text_content[:100] + "..." if len(text_content) > 100 else text_content,
                    "단어": result["original"],
                    "원형": result["base_form"],
                    "품사": "",
                    "뜻(한글)": result["meaning"],
                    "뜻(영어)": "",
                    "동의어": synonyms,
                    "반의어": antonyms,
                    "문맥": result["context"],
                    "분리형여부": result.get("is_separated", False),
                    "신뢰도": result.get("confidence", 0.95),
                    "사용자DB매칭": result.get("user_db_match", True),
                    "매칭방식": result.get("match_type", "사용자DB숙어"),
                    "패턴정보": f"Studio: {passage_info.get('studio_title', '')}, Unit: {passage_info.get('textbook_unit_id', '')}",
                    "문맥적의미": "",
                    "동의어신뢰도": 0.8 if synonyms else 0,
                    "처리방식": "",
                    "포함이유": f"지문 '{passage_info.get('textbook_studio_passage_title', 'Unknown')}' 에서 추출"
                }
                results.append(vocab_entry)

            # 2. 사용자 DB 단일 단어 추출
            user_word_results = self._extract_user_single_words(text_content)
            
            for result in user_word_results:
                # 한글 뜻 생성
                if result['meaning'] == f"{result['base_form']}의 의미":
                    result['meaning'] = self.enhanced_korean_definition(
                        result['base_form'], 
                        result['context'], 
                        is_phrase=False
                    )
                
                # 동의어 생성
                synonyms, antonyms = self.generate_synonyms_and_antonyms(
                    result['base_form'], 
                    result['context']
                )
                
                vocab_entry = {
                    "교재ID": passage_info.get("textbook_id"),
                    "교재명": passage_info.get("book_title", "Advanced Vocabulary"),
                    "지문ID": passage_info.get("textbook_studio_passage_id", text_id),
                    "순서": len(results) + 1,
                    "지문": text_content[:100] + "..." if len(text_content) > 100 else text_content,
                    "단어": result["original"],
                    "원형": result["base_form"],
                    "품사": self._get_pos_from_context(result['base_form'], result['context']),
                    "뜻(한글)": result["meaning"],
                    "뜻(영어)": "",
                    "동의어": synonyms,
                    "반의어": antonyms,
                    "문맥": result["context"],
                    "분리형여부": False,
                    "신뢰도": result.get("confidence", 1.0),
                    "사용자DB매칭": result.get("user_db_match", True),
                    "매칭방식": result.get("match_type", "사용자DB단어"),
                    "패턴정보": f"Studio: {passage_info.get('studio_title', '')}, Unit: {passage_info.get('textbook_unit_id', '')}",
                    "문맥적의미": "",
                    "동의어신뢰도": 0.8 if synonyms else 0,
                    "처리방식": "",
                    "포함이유": f"지문 '{passage_info.get('textbook_studio_passage_title', 'Unknown')}' 에서 추출"
                }
                results.append(vocab_entry)

            return results

        except Exception as e:
            print(f"❌ 텍스트 처리 실패: {e}")
            return results

    def _extract_user_single_words(self, text_content):
        """사용자 DB 단일 단어 추출"""
        results = []
        text_str = force_extract_text(text_content)
        found_positions = set()

        for word in self.user_single_words:
            import re
            pattern = r"\b" + re.escape(word) + r"\b"
            matches = re.finditer(pattern, text_str, re.IGNORECASE)

            for match in matches:
                start, end = match.span()
                
                # 위치 중복 확인
                if any(abs(start - pos[0]) < 3 for pos in found_positions):
                    continue

                context = get_sentence_context(text_str, start, end)
                original_text = text_str[start:end]

                results.append({
                    "original": original_text,
                    "base_form": word,
                    "meaning": f"{word}의 의미",  # 나중에 AI로 대체
                    "context": context,
                    "type": "user_db_word",
                    "confidence": 1.0,
                    "user_db_match": True,
                    "match_type": "사용자DB단어",
                })
                
                found_positions.add((start, end))

        return results

    def _get_pos_from_context(self, word, context):
        """문맥에서 품사 추출"""
        try:
            doc = self.nlp(context)
            for token in doc:
                if token.lemma_.lower() == word.lower():
                    return get_simple_pos(token.pos_)
            return ""
        except:
            return ""
        
