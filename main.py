# main.py - 메인 실행 파일

import argparse
import sys
import pandas as pd
from config import DEFAULT_SETTINGS, FilePaths
from main_extractor import AdvancedVocabExtractor
from utils import safe_read_csv, find_text_column
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ 환경변수 로드 완료")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ API Key 확인: {api_key[:10]}...")
    else:
        print("❌ API Key 로드 실패")
except Exception as e:
    print(f"⚠️ 환경변수 로드 오류: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="고급 패턴 분석 단어장 생성기 v4.0")
    parser.add_argument(
        "--input", "-i", default=FilePaths.DEFAULT_INPUT, help="입력 CSV 파일"
    )
    parser.add_argument(
        "--output", "-o", default=FilePaths.DEFAULT_OUTPUT, help="출력 엑셀 파일"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=2.5, help="난이도 임계값"
    )
    parser.add_argument("--cache", "-c", default="gpt_cache.json", help="GPT 캐시 파일")
    parser.add_argument(
        "--max-tokens", "-mt", type=int, default=200000, help="최대 토큰 사용량"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="상세한 로그 출력")
    parser.add_argument("--user-words", help="사용자 단어 파일 (CSV/XLSX)")
    parser.add_argument(
        "--data-dir", default=FilePaths.DATA_DIR, help="숙어 데이터 디렉토리"
    )
    parser.add_argument(
        "--no-quality-check", action="store_true", help="품질 검사 건너뛰기"
    )

    args = parser.parse_args()

    settings = {
        "DIFFICULTY_THRESHOLD": args.threshold,
        "CACHE_FILE": args.cache,
        "MAX_TOKENS": args.max_tokens,
        "USE_CACHE": True,
        "USE_INTEGRATED_CONTEXTUAL": True,
        "USE_INTEGRATED_DIFFICULTY": True,
        "data_dir": args.data_dir,
    }

    print(f"🚀 고급 패턴 분석 단어장 생성기 v4.0")
    print(f"   • 사용자 DB 숙어 우선 인식: ✅")
    print(f"   • 사용자 DB 단어 우선 포함: ✅")
    print(f"   • 문법 패턴 분석: ✅ (V-ing)")
    print(f"   • 고급 구동사 분석: ✅ (연속형/분리형 자동 구분)")
    print(f"   • 분리형 표시 개선: ✅ (pick ~ up, spend time V-ing)")
    print(f"   • 통합 컨텍스트 의미: ✅")
    print(f"   • 통합 난이도 분석: ✅")

    # 🔥 data_loader를 직접 사용하여 텍스트 로드
    from data_loader import load_texts_data

    try:
        extractor = AdvancedVocabExtractor(
            user_words_file=(
                args.user_words if args.user_words else FilePaths.USER_WORDS_FILE
            ),
            settings=settings,
            csv_file=args.input,
            verbose=args.verbose,
        )

        if not hasattr(extractor, "passage_db") or not extractor.passage_db:
            print(f"❌ 지문 데이터 로딩 실패")
            return

        texts_with_metadata = extractor.passage_db  # 메타데이터 포함된 텍스트 리스트
        print(f"✅ 메타데이터 포함 텍스트 {len(texts_with_metadata)}개 로드 완료")

        print(
            f"📚 총 {len(texts_with_metadata)}개 텍스트에서 고급 패턴 분석 단어장 생성 시작"
        )

        df = extractor.generate_vocabulary_workbook(
            texts_with_metadata,  # ✅ 메타데이터 포함된 텍스트 전달
            output_file=args.output,
            enable_quality_check=not args.no_quality_check,
        )
        # 결과 출력
        if df is not None and len(df) > 0:
            print(f"\n🎉 고급 패턴 분석 단어장 생성 완료!")
            print(f"   📁 파일: {args.output}")
            print(f"   📊 총 항목: {len(df)}개")
            print(
                f"   ✨ 고급 패턴 분석 적용 (분리형: pick ~ up, 문법패턴: spend time V-ing)"
            )

            # 사용자 DB 매칭 통계 출력
            if "사용자DB매칭" in df.columns:
                user_matched = df["사용자DB매칭"].sum()
                total_items = len(df)
                match_ratio = (
                    (user_matched / total_items * 100) if total_items > 0 else 0
                )
                print(f"\n👤 사용자 DB 매칭 결과:")
                print(f"   • 매칭된 항목: {user_matched}개")
                print(f"   • 전체 대비 비율: {match_ratio:.1f}%")

            # 패턴별 통계 출력
            if "매칭방식" in df.columns:
                pattern_stats = df["매칭방식"].value_counts()
                print(f"\n📊 패턴별 추출 통계:")
                for pattern, count in pattern_stats.items():
                    if pattern:
                        print(f"   • {pattern}: {count}개")

        else:
            print("⚠️ 단어장 생성에 실패했거나 추출된 항목이 없습니다.")

    except FileNotFoundError:
        print(f"❌ '{args.input}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


def show_help():
    """도움말 출력"""
    print(
        """
고급 패턴 분석 단어장 생성기 v4.0 사용법:

기본 사용법:
python main.py --input 지문DB.csv --output vocabulary.xlsx

🔥 주요 개선사항 (v4.0):
✅ 고급 구동사 패턴 분석
• 연속형 가능: pick up (그대로 표시)
• 분리 필수: pick ~ up (~ 표시)
• 실제 분리: pick something up → pick ~ up

✅ 문법 패턴 자동 인식
• V-ing 패턴: spend time reading → spend time V-ing
• N V-ing 패턴: prevent him from going → prevent N from V-ing

✅ 사용자 DB 우선 처리
• 사용자 DB 숙어 최우선 인식
• 사용자 DB 단어 우선 포함
• 매칭 방식별 상세 통계

✅ 패턴별 정밀 분석
• 위치 기반 중복 방지
• 문법적 검증 강화
• 신뢰도 기반 우선순위

패턴 표시 예시:
- pick up → pick up (연속형 가능)
- pick something up → pick ~ up (분리 필수)
- spend time reading → spend time V-ing (문법 패턴)

주요 옵션:
--input: 입력 CSV 파일 (기본: 지문DB.csv)
--output: 출력 Excel 파일 (기본: vocabulary_advanced.xlsx)
--user-words: 사용자 단어 파일 (기본: 단어DB.csv)
--data-dir: 참조 숙어 데이터 디렉토리 (기본: data)
--verbose: 상세한 로그 출력
--no-quality-check: 품질 검사 건너뛰기

사용 예시:
python main.py --input 지문DB.csv --output my_vocab.xlsx --verbose
"""
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        show_help()
        sys.exit(0)

    try:
        print("🚀 고급 패턴 분석 단어장 생성기 시작...")
        print("🔥 v4.0 - 분리형/문법패턴 고급 분석 버전")
        main()
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
