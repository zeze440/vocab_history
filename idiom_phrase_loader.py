import pandas as pd
import requests
import json
import logging
from typing import Dict, List, Optional, Union
import time
from requests.exceptions import RequestException
import os
import pickle
import hashlib
from datetime import datetime, timedelta

# 환경 변수로 로그 레벨 제어
if not os.getenv("DEBUG_OPENAI"):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("idiom_loader")


class DatasetMetadata:
    """데이터셋 메타데이터를 저장하는 클래스"""

    def __init__(self, name: str, url: str, source: str, load_date: str = None):
        self.name = name
        self.url = url
        self.source = source
        self.load_date = load_date or time.strftime("%Y-%m-%d %H:%M:%S")
        self.record_count = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "url": self.url,
            "source": self.source,
            "load_date": self.load_date,
            "record_count": self.record_count,
        }


class DataCache:
    """데이터 캐싱을 관리하는 클래스"""

    def __init__(self, cache_dir: str = ".cache", cache_validity: int = None):
        """
        캐시 관리자 초기화
        Args:
            cache_dir: 캐시 파일을 저장할 디렉토리
            cache_validity: 캐시 유효 기간(일)
        """
        self.cache_dir = cache_dir
        self.cache_validity = cache_validity  # 캐시 유효 기간(일)

        # 캐시 디렉토리가 없으면 생성
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"캐시 디렉토리 생성: {cache_dir}")

    def _get_cache_path(self, url: str) -> str:
        """URL에 해당하는 캐시 파일 경로 반환"""
        # URL의 해시값을 파일명으로 사용
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.pkl")

    def get_cache_summary(self) -> Dict:
        """캐시 요약 정보 반환"""
        summary = {"total_files": 0, "total_records": 0, "cache_details": []}

        if not os.path.exists(self.cache_dir):
            return summary

        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pkl"):
                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(cache_path, "rb") as f:
                        cached_data = pickle.load(f)
                        record_count = len(cached_data)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))

                        summary["total_files"] += 1
                        summary["total_records"] += record_count
                        summary["cache_details"].append(
                            {
                                "file": filename,
                                "records": record_count,
                                "modified": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                except Exception as e:
                    logger.warning(f"캐시 파일 읽기 실패 {filename}: {e}")

        return summary

    def get(self, url: str) -> Optional[pd.DataFrame]:
        """
        캐시에서 데이터 가져오기

        Args:
            url: 데이터 소스 URL

        Returns:
            캐시된 데이터프레임 또는 None(캐시 없음)
        """
        cache_path = self._get_cache_path(url)

        if not os.path.exists(cache_path):
            logger.debug(f"캐시 없음: {url}")
            return None

        try:
            # 캐시 파일의 수정 시간 확인
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            now = datetime.now()

            # 캐시 유효기간 확인
            if now - mod_time > timedelta(days=self.cache_validity):
                logger.info(
                    f"캐시 만료됨: {url} (생성일: {mod_time.strftime('%Y-%m-%d')})"
                )
                return None

            # 캐시에서 데이터 로드
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                logger.info(f"캐시에서 로드됨: {url} (레코드 수: {len(cached_data)})")
                return cached_data

        except Exception as e:
            logger.warning(f"캐시 읽기 오류: {str(e)}")
            return None

    def save(self, url: str, df: pd.DataFrame) -> bool:
        """
        데이터를 캐시에 저장

        Args:
            url: 데이터 소스 URL
            df: 저장할 데이터프레임

        Returns:
            성공 여부
        """
        cache_path = self._get_cache_path(url)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
            logger.info(f"캐시에 저장됨: {url} (레코드 수: {len(df)})")
            return True
        except Exception as e:
            logger.warning(f"캐시 저장 오류: {str(e)}")
            return False


# 전역 캐시 관리자 인스턴스 생성
cache = DataCache(cache_validity=9999)


def safe_request(
    url: str, stream: bool = False, timeout: int = 30, max_retries: int = 3
) -> requests.Response:
    """안전하게 HTTP 요청을 처리하는 함수"""
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, stream=stream, timeout=timeout)
            response.raise_for_status()  # 4XX, 5XX 오류 검출
            return response
        except RequestException as e:
            retries += 1
            wait_time = 2**retries  # 지수 백오프
            logger.warning(
                f"요청 실패 ({retries}/{max_retries}): {str(e)}. {wait_time}초 후 재시도..."
            )
            time.sleep(wait_time)

    raise RequestException(f"최대 재시도 횟수({max_retries})를 초과했습니다: {url}")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """데이터프레임의 유효성을 검사하고 필요한 열이 없으면 추가"""
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    return df


def load_tatoeba_phrases(
    limit: int = 100000,  # 기본값을 10만으로 증가
    chunk_size: int = 1000,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Tatoeba에서 영어 문장 데이터 로드 (청크 단위로 처리)"""
    metadata = DatasetMetadata(
        name="Tatoeba Phrases",
        url=f"https://downloads.tatoeba.org/exports/sentences.csv?limit={limit}",
        source="tatoeba",
    )

    logger.info(f"{metadata.name} 데이터 로드 중... (최대 {limit}개, 캐시 사용 안함)")

    try:
        # 실제 Tatoeba sentences.csv 다운로드 URL
        url = "https://downloads.tatoeba.org/exports/sentences.csv"
        res = safe_request(url, stream=True)

        phrases = []
        processed_count = 0  # 처리된 라인 수
        valid_phrases = 0  # 유효한 구문 수
        skip_header = True  # 헤더 스킵 플래그

        print(f"📥 Tatoeba 데이터 처리 중... (최대 {limit:,}개)")

        for line in res.iter_lines():
            # 헤더 스킵 (첫 번째 라인)
            if skip_header:
                skip_header = False
                continue

            # 원하는 개수만큼 수집했으면 중단
            if valid_phrases >= limit:
                break

            try:
                # TSV 파싱 (탭으로 구분)
                row = line.decode("utf-8").split("\t")

                # 최소 3개 컬럼 필요: ID, 언어코드, 문장
                if len(row) >= 3 and row[1] == "eng":  # 영어만 필터링
                    sentence = row[2].strip()

                    # 빈 문장 스킵
                    if not sentence:
                        continue

                    words = sentence.split()

                    # 문장 길이 필터링 (원하는 조건으로 수정 가능)
                    if 2 <= len(words) <= 15:  # 2-15단어로 범위 확대
                        phrases.append(
                            {
                                "phrase": sentence,
                                "definition": "",
                                "example": "",
                                "source": metadata.source,
                            }
                        )
                        valid_phrases += 1

            except (UnicodeDecodeError, IndexError) as e:
                # 디코딩 오류나 컬럼 부족 무시
                continue
            except Exception as e:
                # 기타 오류 무시하고 계속 진행
                continue

            processed_count += 1

            # 진행 상황 출력 (10,000개마다)
            if processed_count % 10000 == 0:
                print(
                    f"  📊 처리됨: {processed_count:,}줄, 유효 구문: {valid_phrases:,}개"
                )

        print(
            f"✅ Tatoeba 처리 완료: 총 {processed_count:,}줄 처리, {valid_phrases:,}개 구문 수집"
        )

        df = pd.DataFrame(phrases)
        metadata.record_count = len(df)

        logger.info(
            f"{metadata.name} 데이터 로드 완료: {metadata.record_count}개 레코드"
        )

        # 메타데이터 추가
        df.attrs["metadata"] = metadata.to_dict()

        # 캐시에 저장
        if use_cache:
            cache.save(metadata.url, df)

        return df

    except Exception as e:
        logger.error(f"{metadata.name} 로드 중 오류 발생: {str(e)}")
        return pd.DataFrame(columns=["phrase", "definition", "example", "source"])


def load_all_idioms_and_phrases(
    use_cache: bool = True,
    save_to_csv: Optional[str] = None,
    tatoeba_limit: int = 30000,
) -> pd.DataFrame:
    master_cache_file = os.path.join(cache.cache_dir, "master_idioms_combined.pkl")

    # ✅ 로컬 CSV 파일 먼저 로드
    local_csv_path = "all_merged_idioms.csv"
    custom_idioms = []

    if os.path.exists(local_csv_path):
        try:
            local_df = pd.read_csv(local_csv_path)
            if "idiom" in local_df.columns:
                custom_idioms = (
                    local_df["idiom"].dropna().str.strip().str.lower().tolist()
                )
                print(f"📖 {local_csv_path}에서 {len(custom_idioms)}개 숙어 로드됨")
            else:
                print(f"⚠️ {local_csv_path}에 'idiom' 컬럼이 없습니다")
        except Exception as e:
            print(f"❌ {local_csv_path} 로드 실패: {e}")

    # 캐시 확인 (로컬 숙어 포함된 완전한 캐시)
    if use_cache and os.path.exists(master_cache_file):
        try:
            with open(master_cache_file, "rb") as f:
                combined_df = pickle.load(f)

            # 로컬 숙어가 포함되어 있는지 확인
            if len(combined_df) > 15000:  # Tatoeba(10K) + 로컬(24K) 정도면 포함된 것
                logger.info(
                    f"✅ 캐시에서 완전한 숙어 데이터 로드: {len(combined_df)}개"
                )
                return combined_df
            else:
                logger.info("캐시가 불완전함. 새로 생성...")
        except Exception as e:
            logger.warning(f"캐시 로드 실패, 새로 생성: {e}")

    # 새로 생성
    logger.info("영어 숙어 및 구문 데이터를 새로 로드합니다...")
    tatoeba_df = load_tatoeba_phrases(limit=tatoeba_limit, use_cache=False)
    combined_df = pd.concat([tatoeba_df], ignore_index=True)

    # 로컬 숙어 병합
    if custom_idioms:
        df_local = pd.DataFrame(
            {"phrase": custom_idioms, "definition": "", "example": "", "source": "user"}
        )
        combined_df = pd.concat([combined_df, df_local], ignore_index=True)

    combined_df.drop_duplicates(subset=["phrase"], keep="first", inplace=True)

    # 캐시 저장
    if use_cache:
        try:
            with open(master_cache_file, "wb") as f:
                pickle.dump(combined_df, f)
            logger.info(f"✅ 통합 캐시 저장 완료: {master_cache_file}")
        except Exception as e:
            logger.error(f"통합 캐시 저장 실패: {e}")

    return combined_df


# 메인 함수 (스크립트로 실행될 때 사용)
def main():
    import argparse

    parser = argparse.ArgumentParser(description="영어 숙어 및 구문 데이터 로더")
    parser.add_argument("--output", "-o", type=str, help="결과를 저장할 CSV 파일 경로")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=30000,
        help="Tatoeba에서 가져올 최대 레코드 수",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로깅 활성화")
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용하지 않음")
    parser.add_argument(
        "--clear-cache", action="store_true", help="캐시 초기화 후 실행"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=".cache", help="캐시 디렉토리 경로"
    )
    parser.add_argument(
        "--cache-validity", type=int, default=100, help="캐시 유효 기간(일)"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 캐시 설정
    global cache
    cache = DataCache(cache_dir=args.cache_dir, cache_validity=args.cache_validity)

    # 캐시 초기화 옵션 처리
    if args.clear_cache:
        import shutil

        try:
            shutil.rmtree(args.cache_dir)
            os.makedirs(args.cache_dir)
            logger.info(f"캐시 디렉토리 초기화: {args.cache_dir}")
        except Exception as e:
            logger.error(f"캐시 초기화 오류: {str(e)}")

    df = load_all_idioms_and_phrases(
        use_cache=not args.no_cache, save_to_csv=args.output, tatoeba_limit=args.limit
    )

    print(f"\n총 {len(df)}개의 영어 숙어와 구문을 로드했습니다.")
    print(f"소스별 분포:")
    for source, count in df["source"].value_counts().items():
        print(f"- {source}: {count}개")


if __name__ == "__main__":
    main()
