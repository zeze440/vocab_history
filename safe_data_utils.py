"""
안전한 데이터 처리 유틸리티 함수들
improved_vocab_extractor.py에서 사용하는 전역 함수 래퍼들
"""

import pandas as pd
from typing import Any, Union


class SafeDataHandler:
    """데이터 타입 안전성을 보장하는 핸들러"""

    @staticmethod
    def safe_get_column_value(
        row: pd.Series, column: str, default: str = "", expected_type: type = str
    ) -> Any:
        """안전하게 컬럼 값 가져오기 (타입 검증 포함)"""
        try:
            if column not in row.index:
                return SafeDataHandler._convert_to_type(default, expected_type)

            value = row[column]

            # NaN, None 체크
            if pd.isna(value) or value is None:
                return SafeDataHandler._convert_to_type(default, expected_type)

            # 타입 변환 및 검증
            return SafeDataHandler._convert_to_type(value, expected_type)

        except Exception as e:
            print(f"⚠️ 컬럼 '{column}' 값 추출 실패: {e}")
            return SafeDataHandler._convert_to_type(default, expected_type)

    @staticmethod
    def safe_string_operation(text: Any, operation: str, *args, **kwargs) -> str:
        """안전한 문자열 작업"""
        try:
            if text is None or pd.isna(text):
                return ""

            text_str = str(text).strip()

            if operation == "lower":
                return text_str.lower()
            elif operation == "upper":
                return text_str.upper()
            elif operation == "replace":
                if len(args) >= 2:
                    return text_str.replace(args[0], args[1])
            elif operation == "split":
                delimiter = args[0] if args else " "
                return text_str.split(delimiter)
            elif operation == "strip":
                chars = args[0] if args else None
                return text_str.strip(chars)
            elif operation == "find":
                if args:
                    return text_str.find(args[0])
            elif operation == "startswith":
                if args:
                    return text_str.startswith(args[0])
            elif operation == "endswith":
                if args:
                    return text_str.endswith(args[0])

            return text_str

        except Exception as e:
            print(f"⚠️ 문자열 작업 '{operation}' 실패: {e}")
            return str(text) if text is not None else ""

    @staticmethod
    def safe_numeric_operation(
        value: Any, operation: str, default: Union[int, float] = 0
    ) -> Union[int, float]:
        """안전한 숫자 작업"""
        try:
            if value is None or pd.isna(value):
                return default

            if operation == "int":
                return int(float(value))
            elif operation == "float":
                return float(value)
            elif operation == "round":
                decimal_places = 2  # 기본값
                return round(float(value), decimal_places)

            return float(value)

        except Exception as e:
            print(f"⚠️ 숫자 작업 '{operation}' 실패: {e}")
            return default

    @staticmethod
    def _convert_to_type(value: Any, expected_type: type) -> Any:
        """값을 지정된 타입으로 안전하게 변환"""
        try:
            if expected_type == str:
                if value is None or pd.isna(value):
                    return ""
                return str(value).strip()

            elif expected_type == int:
                if value is None or pd.isna(value):
                    return 0
                return int(float(value))

            elif expected_type == float:
                if value is None or pd.isna(value):
                    return 0.0
                return float(value)

            elif expected_type == bool:
                if value is None or pd.isna(value):
                    return False
                if isinstance(value, str):
                    return value.lower() in ["true", "1", "yes", "예", "y"]
                return bool(value)

            elif expected_type == list:
                if value is None or pd.isna(value):
                    return []
                if isinstance(value, str):
                    return [item.strip() for item in value.split(",") if item.strip()]
                elif isinstance(value, (list, tuple)):
                    return list(value)
                else:
                    return [value]
            else:
                return value

        except Exception as e:
            print(f"⚠️ 타입 변환 실패 ({value} -> {expected_type}): {e}")
            if expected_type == str:
                return ""
            elif expected_type == int:
                return 0
            elif expected_type == float:
                return 0.0
            elif expected_type == bool:
                return False
            elif expected_type == list:
                return []
            else:
                return None


# 전역 함수 래퍼들
def safe_get_column_value(row, column, default="", expected_type=str):
    """전역 함수 래퍼 - SafeDataHandler.safe_get_column_value"""
    return SafeDataHandler.safe_get_column_value(row, column, default, expected_type)


def safe_string_operation(text, operation, *args, **kwargs):
    """전역 함수 래퍼 - SafeDataHandler.safe_string_operation"""
    return SafeDataHandler.safe_string_operation(text, operation, *args, **kwargs)


def safe_numeric_operation(value, operation, default=0):
    """전역 함수 래퍼 - SafeDataHandler.safe_numeric_operation"""
    return SafeDataHandler.safe_numeric_operation(value, operation, default)
