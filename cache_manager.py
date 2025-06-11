# cache_manager.py - 캐시 관리 시스템

import os
import json
import time
import hashlib
from typing import Dict, Any
from utils import load_json_safe, save_json_safe


class SafeCacheManager:
    """Windows 호환 간단한 캐시 관리자"""

    def __init__(self, cache_dir: str = "cache", app_name: str = "vocab_extractor"):
        self.cache_dir = cache_dir
        self.app_name = app_name
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_filename(self, cache_type: str, identifier: str = "") -> str:
        """캐시 파일명 생성"""
        if identifier:
            hash_id = hashlib.md5(identifier.encode()).hexdigest()[:8]
            filename = f"{self.app_name}_{cache_type}_{hash_id}.json"
        else:
            filename = f"{self.app_name}_{cache_type}.json"
        return os.path.join(self.cache_dir, filename)

    def load_cache(self, cache_type: str, identifier: str = "") -> Dict[str, Any]:
        """캐시 로드"""
        cache_file = self.get_cache_filename(cache_type, identifier)
        cache_data = load_json_safe(cache_file, {})

        if isinstance(cache_data, dict) and "data" in cache_data:
            print(f"✅ 캐시 로드 성공: {cache_type}")
            return cache_data["data"]
        elif isinstance(cache_data, dict):
            return cache_data
        return {}

    def save_cache(self, cache_type: str, data: Dict[str, Any], identifier: str = ""):
        """캐시 저장"""
        cache_file = self.get_cache_filename(cache_type, identifier)

        cache_data = {
            "metadata": {
                "created_at": time.time(),
                "app_name": self.app_name,
                "cache_type": cache_type,
            },
            "data": data,
        }

        if save_json_safe(cache_data, cache_file):
            print(f"✅ 캐시 저장 성공: {cache_type}")
        else:
            print(f"⚠️ 캐시 저장 실패: {cache_type}")

    def clear_cache(self, cache_type: str = None):
        """캐시 삭제"""
        try:
            if cache_type:
                cache_file = self.get_cache_filename(cache_type)
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    print(f"✅ 캐시 삭제: {cache_type}")
            else:
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith(self.app_name):
                        os.remove(os.path.join(self.cache_dir, filename))
                print(f"✅ 모든 캐시 삭제 완료")
        except Exception as e:
            print(f"⚠️ 캐시 삭제 실패: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        try:
            cache_files = [
                f for f in os.listdir(self.cache_dir) if f.startswith(self.app_name)
            ]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files
            )

            return {
                "file_count": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": self.cache_dir,
                "files": cache_files,
            }
        except Exception as e:
            print(f"⚠️ 캐시 통계 조회 실패: {e}")
            return {}


class UnifiedCacheManager:
    """improved_vocab_extractor.py와 quality_checker.py에서 공통 사용할 캐시 관리자"""

    def __init__(self):
        self.cache_manager = SafeCacheManager(
            cache_dir="cache", app_name="vocab_system"
        )

        # 캐시 타입 정의
        self.CACHE_TYPES = {
            "gpt_difficulty": "gpt_difficulty",
            "gpt_quality": "gpt_quality",
            "gpt_contextual": "gpt_contextual",
            "separable_analysis": "separable_analysis",
            "user_words": "user_words",
            "easy_words": "easy_words",
        }

    def load_gpt_cache(
        self, cache_type: str, user_identifier: str = ""
    ) -> Dict[str, Any]:
        """GPT 캐시 로드 (타입별)"""
        return self.cache_manager.load_cache(cache_type, user_identifier)

    def save_gpt_cache(
        self, cache_type: str, data: Dict[str, Any], user_identifier: str = ""
    ):
        """GPT 캐시 저장 (타입별)"""
        self.cache_manager.save_cache(cache_type, data, user_identifier)

    def merge_caches(
        self, cache_type: str, new_data: Dict[str, Any], user_identifier: str = ""
    ):
        """기존 캐시와 새 데이터 병합"""
        try:
            existing_cache = self.load_gpt_cache(cache_type, user_identifier)

            # 기존 데이터와 병합 (새 데이터가 우선)
            merged_data = {**existing_cache, **new_data}

            self.save_gpt_cache(cache_type, merged_data, user_identifier)

            print(
                f"✅ 캐시 병합 완료: {cache_type} (기존 {len(existing_cache)}개 + 신규 {len(new_data)}개 = 총 {len(merged_data)}개)"
            )

        except Exception as e:
            print(f"⚠️ 캐시 병합 실패 ({cache_type}): {e}")

    def cleanup_old_caches(self, days_old: int = 7):
        """오래된 캐시 정리"""
        try:
            from datetime import datetime, timedelta

            cutoff_date = datetime.now() - timedelta(days=days_old)
            removed_count = 0

            for filename in os.listdir(self.cache_manager.cache_dir):
                if filename.startswith(
                    self.cache_manager.app_name
                ) and filename.endswith(".json"):
                    filepath = os.path.join(self.cache_manager.cache_dir, filename)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

                    if file_mtime < cutoff_date:
                        os.remove(filepath)
                        removed_count += 1

            if removed_count > 0:
                print(f"✅ 오래된 캐시 {removed_count}개 삭제 완료 ({days_old}일 이상)")

        except Exception as e:
            print(f"⚠️ 캐시 정리 실패: {e}")

    def get_unified_stats(self) -> Dict[str, Any]:
        """통합 캐시 통계"""
        return self.cache_manager.get_cache_stats()


# 전역 캐시 관리자 인스턴스
_global_cache_manager = None


def get_cache_manager() -> UnifiedCacheManager:
    """전역 캐시 관리자 인스턴스 반환 (싱글톤 패턴)"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = UnifiedCacheManager()
    return _global_cache_manager
