# data_loader.py - 데이터 로딩 유틸리티

import os
import glob
import pandas as pd
from utils import safe_read_csv


def find_text_column(df):
    """텍스트 컬럼을 자동으로 찾는 함수 - 수정된 버전"""

    # 🔥 우선순위별 컬럼명 목록 (content 추가)
    priority_columns = [
        "content",  # 🔥 새로 추가 - 가장 높은 우선순위
        "text",
        "passage",
        "content_text",
        "paragraph",
        "article",
        "body",
        "description",
        "story",
        "passage_text",
    ]

    # 대소문자 구분 없이 검색
    columns_lower = [col.lower() for col in df.columns]

    # 우선순위별로 검색
    for target in priority_columns:
        if target.lower() in columns_lower:
            # 실제 컬럼명 찾기
            actual_column = df.columns[columns_lower.index(target.lower())]
            print(f"✅ 텍스트 컬럼 발견: '{actual_column}'")
            return actual_column

    # 🔥 부분 매치 검색 (더 유연한 검색)
    for col in df.columns:
        col_lower = col.lower()
        if any(
            keyword in col_lower for keyword in ["content", "text", "passage", "body"]
        ):
            print(f"✅ 텍스트 컬럼 발견 (부분매치): '{col}'")
            return col

    # 🔥 데이터 길이 기반 추측
    text_candidates = []
    for col in df.columns:
        if df[col].dtype == "object":  # 문자열 컬럼만
            # 평균 텍스트 길이 계산
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 50:  # 평균 50자 이상이면 텍스트일 가능성
                text_candidates.append((col, avg_length))

    if text_candidates:
        # 가장 긴 평균 길이를 가진 컬럼 선택
        best_column = max(text_candidates, key=lambda x: x[1])[0]
        print(
            f"✅ 텍스트 컬럼 추정: '{best_column}' (평균 길이: {max(text_candidates, key=lambda x: x[1])[1]:.1f}자)"
        )
        return best_column

    # 마지막 시도: 첫 번째 object 타입 컬럼
    for col in df.columns:
        if df[col].dtype == "object":
            print(f"⚠️ 기본 텍스트 컬럼 사용: '{col}'")
            return col

    raise ValueError("텍스트 컬럼을 찾을 수 없습니다")


def load_user_words_data(file_path):
    """사용자 단어 데이터 로딩 - 기존과 동일"""
    print(f"📚 사용자 단어 데이터 로딩: {file_path}")

    if not os.path.exists(file_path):
        print(f"⚠️ 사용자 단어 파일이 없습니다: {file_path}")
        return [], []

    try:
        df = safe_read_csv(file_path)

        if df is None or df.empty:
            print("⚠️ 빈 사용자 단어 파일입니다")
            return [], []

        # 첫 번째 컬럼을 단어로 사용
        word_column = df.columns[0]
        words = df[word_column].dropna().astype(str).tolist()

        # 단어와 숙어 분리
        single_words = [w.strip() for w in words if w.strip() and " " not in w.strip()]
        idioms = [w.strip() for w in words if w.strip() and " " in w.strip()]

        print(f"   ✅ 사용자 단어: {len(single_words)}개")
        print(f"   ✅ 사용자 숙어: {len(idioms)}개")

        return single_words, idioms

    except Exception as e:
        print(f"❌ 사용자 단어 데이터 로딩 실패: {e}")
        return [], []


def load_texts_data(file_path):
    """지문 데이터 로딩 - 수정된 버전 (지문 DB 정보 포함)"""
    print(f"📖 지문 데이터 로딩: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    try:
        # CSV 파일 읽기
        df = safe_read_csv(file_path)

        if df is None or df.empty:
            raise ValueError("빈 데이터파일입니다")

        print(f"   📊 전체 행 수: {len(df)}")
        print(f"   📊 컬럼 수: {len(df.columns)}")
        print(f"   📊 컬럼들: {df.columns.tolist()}")

        # 텍스트 컬럼 찾기
        text_column = find_text_column(df)

        # 🔥 유효한 텍스트만 필터링하되 지문 DB 정보 모두 포함
        valid_texts = []
        for idx, row in df.iterrows():
            text_content = row[text_column]

            # None, NaN, 빈 문자열 체크
            if pd.isna(text_content) or not str(text_content).strip():
                continue

            text_str = str(text_content).strip()

            # 최소 길이 체크 (10자 이상)
            if len(text_str) < 10:
                continue

            # 🔥 ID 생성 (textbook_studio_passage_id 우선 사용)
            if "textbook_studio_passage_id" in df.columns and pd.notna(
                row["textbook_studio_passage_id"]
            ):
                text_id = str(row["textbook_studio_passage_id"])
            elif "textbook_id" in df.columns:
                text_id = f"text_{row['textbook_id']}"
            else:
                text_id = f"text_{idx + 1}"

            # 🔥 지문 DB의 모든 정보를 포함하여 반환
            text_info = {
                "id": text_id,
                "content": text_str,
                "original_index": idx,
                # 🔥 지문 DB 메타데이터 추가
                "textbook_id": row.get("textbook_id", None),
                "product_id": row.get("product_id", None),
                "textbook_studio_passage_id": row.get(
                    "textbook_studio_passage_id", None
                ),
                "textbook_unit_id": row.get("textbook_unit_id", None),
                "book_title": row.get("book_title", None),
                "studio_title": row.get("studio_title", None),
                "studio_series": row.get("studio_series", None),
                "studio_title2": row.get("studio_title2", None),
                "textbook_studio_passage_title": row.get(
                    "textbook_studio_passage_title", None
                ),
                "passage_order": row.get("passage_order", None),
                # 원본 행 정보도 포함 (필요시 사용)
                "original_row": row.to_dict(),
            }

            valid_texts.append(text_info)

        print(f"   ✅ 유효한 텍스트: {len(valid_texts)}개")

        if not valid_texts:
            raise ValueError("유효한 텍스트가 없습니다")

        return valid_texts

    except Exception as e:
        print(f"❌ 지문 데이터 로딩 실패: {e}")
        raise


def load_custom_idioms_from_data_directory(data_dir: str = "data") -> list:
    """data 디렉토리에서 숙어 로드"""
    idioms = set()
    loading_summary = []

    print(f"📁 data 디렉토리 숙어 로딩 시작: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"❌ 디렉토리가 존재하지 않습니다: {data_dir}")
        return []

    file_patterns = ["*.csv", "*.txt", "*.xlsx", "*.xls"]
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(os.path.join(data_dir, pattern)))

    print(f"🔍 발견된 파일 수: {len(all_files)}개")

    for file_path in all_files:
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        file_idioms_count = 0

        try:
            print(f"📄 처리 중: {filename}")

            if file_ext == ".csv":
                file_idioms_count = _load_csv_idioms(file_path, idioms)
            elif file_ext == ".txt":
                file_idioms_count = _load_txt_idioms(file_path, idioms)
            elif file_ext in [".xlsx", ".xls"]:
                file_idioms_count = _load_excel_idioms(file_path, idioms)

            loading_summary.append(
                {
                    "file": filename,
                    "type": file_ext,
                    "loaded_count": file_idioms_count,
                    "status": "success" if file_idioms_count > 0 else "empty",
                }
            )

        except Exception as e:
            print(f"   ❌ {filename} 처리 실패: {e}")
            loading_summary.append(
                {
                    "file": filename,
                    "type": file_ext,
                    "loaded_count": 0,
                    "status": "failed",
                    "error": str(e),
                }
            )

    final_idioms = sorted(set(idioms))
    print(f"\n📊 data 디렉토리 로딩 완료: 총 {len(final_idioms)}개 숙어")
    return final_idioms


def _load_csv_idioms(file_path, idioms):
    """CSV 파일에서 숙어 로드"""
    df = safe_read_csv(file_path)
    if df is None:
        return 0

    possible_columns = ["phrase", "idiom", "expression", "text", "content", "원형"]
    target_column = next(
        (
            col
            for col in df.columns
            if col.lower() in [c.lower() for c in possible_columns]
        ),
        df.columns[0] if len(df.columns) > 0 else None,
    )

    if target_column and target_column in df.columns:
        phrases = df[target_column].dropna().astype(str).str.strip().str.lower()
        phrases = phrases[phrases != ""].unique()
        valid_phrases = [
            phrase
            for phrase in phrases
            if 2 <= len(phrase.split()) <= 8
            and len(phrase) <= 100
            and phrase.replace(" ", "").replace("-", "").isalpha()
        ]
        idioms.update(valid_phrases)
        file_idioms_count = len(valid_phrases)
        print(f"   ✅ CSV: {file_idioms_count}개 추출 (컬럼: {target_column})")
        return file_idioms_count

    return 0


def _load_txt_idioms(file_path, idioms):
    """TXT 파일에서 숙어 로드"""
    from config import ENCODING_ORDER

    for encoding in ENCODING_ORDER:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            phrases = [
                line.strip().lower()
                for line in lines
                if line.strip()
                and not line.startswith(("#", "//"))
                and 2 <= len(line.split()) <= 8
                and len(line) <= 100
            ]
            idioms.update(phrases)
            file_idioms_count = len(phrases)
            print(f"   ✅ TXT: {file_idioms_count}개 추출 (인코딩: {encoding})")
            return file_idioms_count
        except UnicodeDecodeError:
            continue

    print(f"   ❌ TXT: 모든 인코딩 실패")
    return 0


def _load_excel_idioms(file_path, idioms):
    """Excel 파일에서 숙어 로드"""
    excel_file = pd.ExcelFile(file_path)
    total_phrases = []

    for sheet_name in excel_file.sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            possible_columns = [
                "phrase",
                "idiom",
                "expression",
                "text",
                "content",
                "원형",
            ]
            target_column = next(
                (
                    col
                    for col in df.columns
                    if col.lower() in [c.lower() for c in possible_columns]
                ),
                df.columns[0] if len(df.columns) > 0 else None,
            )
            if target_column and target_column in df.columns:
                phrases = df[target_column].dropna().astype(str).str.strip().str.lower()
                phrases = phrases[phrases != ""].unique()
                valid_phrases = [
                    phrase
                    for phrase in phrases
                    if 2 <= len(phrase.split()) <= 8
                    and len(phrase) <= 100
                    and phrase.replace(" ", "").replace("-", "").isalpha()
                ]
                total_phrases.extend(valid_phrases)
                print(f"      📋 {sheet_name}: {len(valid_phrases)}개")
        except Exception as e:
            print(f"      ❌ {sheet_name} 시트 처리 실패: {e}")

    idioms.update(total_phrases)
    file_idioms_count = len(total_phrases)
    print(f"   ✅ Excel: 총 {file_idioms_count}개 추출")
    return file_idioms_count
