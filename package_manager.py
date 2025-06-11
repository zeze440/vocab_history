# package_manager.py - 패키지 초기화 및 검증

import os
import sys

# 전역 변수 초기화
spacy = None
nlp = None
openai = None
client = None
nltk = None
lemmatizer = None


def safe_import_packages():
    """필수 패키지들을 안전하게 import"""
    global spacy, openai, nltk, nlp, lemmatizer, client

    # 🔥 .env 파일 로드 (맨 위에 추가)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env 파일 로드 완료")
    except ImportError:
        print("⚠️ python-dotenv가 설치되지 않음. pip install python-dotenv")
    except Exception as e:
        print(f"⚠️ .env 로드 실패: {e}")
    # SpaCy 초기화
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        print(
            "❌ spacy가 설치되지 않았습니다. 'pip install spacy' 실행 후 'python -m spacy download en_core_web_sm' 실행하세요."
        )
        return False
    except OSError:
        print(
            "❌ spacy 영어 모델이 없습니다. 'python -m spacy download en_core_web_sm' 실행하세요."
        )
        return False

    # OpenAI 초기화
    try:
        import openai

        # OpenAI API 키 검증
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            print("💡 다음 중 한 가지 방법으로 설정하세요:")
            print("   1. export OPENAI_API_KEY='your-api-key-here'")
            print("   2. .env 파일에 OPENAI_API_KEY=your-api-key-here 추가")
            print("   3. 시스템 환경변수로 설정")
            return False

        try:
            client = openai.OpenAI(api_key=api_key)
            # API 키 유효성 간단 테스트
            test_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            print("✅ OpenAI API 연결 및 인증 성공")
        except openai.AuthenticationError:
            print("❌ OpenAI API 키가 유효하지 않습니다. API 키를 확인해주세요.")
            return False
        except openai.RateLimitError:
            print("⚠️ OpenAI API 사용량 한도에 도달했습니다. 잠시 후 다시 시도하세요.")
            return False
        except openai.APIError as e:
            print(f"❌ OpenAI API 오류: {e}")
            return False
        except Exception as e:
            print(f"❌ OpenAI 초기화 실패: {e}")
            return False

    except ImportError:
        print("❌ openai가 설치되지 않았습니다. 'pip install openai' 실행하세요.")
        return False

    # NLTK 초기화
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()

        # NLTK 데이터 확인
        required_nltk_data = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/wordnet", "wordnet"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/stopwords", "stopwords"),
        ]

        for data_path, download_name in required_nltk_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                print(f"📦 NLTK {download_name} 데이터 다운로드 중...")
                try:
                    nltk.download(download_name, quiet=True)
                except Exception as e:
                    print(f"⚠️ NLTK {download_name} 다운로드 실패: {e}")
                    print("인터넷 연결을 확인하거나 수동으로 다운로드하세요.")

    except ImportError:
        print("❌ nltk가 설치되지 않았습니다. 'pip install nltk' 실행하세요.")
        return False

    return True


def get_nlp_model():
    """SpaCy 모델 반환"""
    global nlp
    return nlp


def get_openai_client():
    """OpenAI 클라이언트 반환"""
    global client
    return client


def get_lemmatizer():
    """NLTK Lemmatizer 반환"""
    global lemmatizer
    return lemmatizer


def initialize_packages():
    """패키지 초기화 실행"""
    if not safe_import_packages():
        print("❌ 필수 패키지 import 실패. 제한된 모드로 계속 진행합니다.")
        return False
    return True
