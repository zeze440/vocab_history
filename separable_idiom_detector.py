# separable_idiom_detector.py - 분리형 숙어 감지 시스템

import re
import json
from typing import Set, Dict, List
from utils import load_json_safe, save_json_safe


class SeparableIdiomDetector:
    """사용자 DB의 분리형 숙어 감지기 (OpenAI 활용)"""

    def __init__(self, client, verbose=False):
        self.client = client
        self.verbose = verbose
        self.separable_cache = {}  # GPT 호출 결과 캐싱
        self.gpt_calls = 0

        # 분리형 숙어 데이터베이스
        self.user_separable_idioms = {}  # {원본숙어: 분리형정보}
        self.separable_patterns = {}  # {원본숙어: [패턴들]}

        print("🔧 사용자 DB 분리형 숙어 감지기 초기화 완료")

    def analyze_user_idioms_with_gpt(self, user_idioms: Set[str]) -> Dict[str, Dict]:
        """GPT로 사용자 DB 숙어들의 분리형 가능성 분석"""
        print(f"🤖 GPT로 사용자 숙어 분리형 분석 시작: {len(user_idioms)}개")

        results = {}

        # 띄어쓰기가 있는 숙어들만 분석
        potential_idioms = [
            idiom for idiom in user_idioms if " " in idiom and len(idiom.split()) == 2
        ]

        print(f"   📋 분석 대상 2단어 숙어: {len(potential_idioms)}개")

        for idiom in potential_idioms:
            if idiom in self.separable_cache:
                results[idiom] = self.separable_cache[idiom]
                continue

            try:
                analysis = self._gpt_analyze_separable_idiom(idiom)
                results[idiom] = analysis
                self.separable_cache[idiom] = analysis

                if self.verbose and analysis.get("is_separable", False):
                    display = analysis.get("display_form", idiom)
                    print(f"      ✅ 분리형 확인: {idiom} → {display}")

            except Exception as e:
                if self.verbose:
                    print(f"      ❌ GPT 분석 실패 ({idiom}): {e}")
                continue

        # 분리형 숙어만 필터링
        separable_count = len(
            [r for r in results.values() if r.get("is_separable", False)]
        )
        print(f"   🎯 분리형 숙어 발견: {separable_count}개")

        return results

    def _gpt_analyze_separable_idiom(self, idiom: str) -> Dict:
        """GPT로 개별 숙어의 분리형 여부 분석"""
        words = idiom.split()
        if len(words) != 2:
            return {"is_separable": False, "reason": "Not a two-word phrase"}

        verb, particle = words[0], words[1]

        prompt = f"""
Analyze the English phrasal verb "{idiom}":

Please determine if this is a separable phrasal verb.

Characteristics of separable phrasal verbs:
1. Structure: verb + particle (up, down, on, off, out, in, away, back, etc.)
2. When there's an object, it can be placed between the verb and particle
   Example: pick up → pick something up, pick it up
3. Pronoun objects (it, him, her, them) MUST be placed between verb and particle
   Example: pick it up (✓), pick up it (✗)

Analyze "{idiom}":
- Is "{verb}" a verb?
- Is "{particle}" a particle?
- Can an object be placed between them?
- Is this actually used as a separable phrasal verb in real English?

Please respond in JSON format:
{{
    "is_separable": true/false,
    "verb": "{verb}",
    "particle": "{particle}", 
    "display_form": "if separable, show as 'verb ~ particle' format",
    "examples": ["examples of separable usage"],
    "reason": "reasoning for the decision",
    "confidence": 0.0-1.0
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in English phrasal verbs. You can accurately identify separable phrasal verbs and their usage patterns.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )

            self.gpt_calls += 1
            content = response.choices[0].message.content.strip()

            # JSON 파싱
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()

            result = json.loads(content)

            # 결과 검증 및 보완
            if not isinstance(result.get("is_separable"), bool):
                result["is_separable"] = False

            if result["is_separable"] and not result.get("display_form"):
                result["display_form"] = f"{verb} ~ {particle}"

            return result

        except Exception as e:
            if self.verbose:
                print(f"GPT 분석 오류: {e}")
            return {
                "is_separable": False,
                "reason": f"GPT 분석 실패: {str(e)}",
                "confidence": 0.0,
            }

    def build_separable_patterns(self, separable_analysis: Dict[str, Dict]):
        """분리형 숙어들의 감지 패턴 생성"""
        print(f"🔧 분리형 숙어 패턴 생성 중...")

        for idiom, analysis in separable_analysis.items():
            if not analysis.get("is_separable", False):
                continue

            words = idiom.split()
            if len(words) != 2:
                continue

            verb, particle = words[0], words[1]
            display_form = analysis.get("display_form", f"{verb} ~ {particle}")

            # 다양한 분리형 패턴들 생성
            patterns = self._generate_detection_patterns(verb, particle)

            self.user_separable_idioms[idiom] = {
                "display_form": display_form,
                "verb": verb,
                "particle": particle,
                "confidence": analysis.get("confidence", 0.9),
                "examples": analysis.get("examples", []),
                "reason": analysis.get("reason", ""),
            }

            self.separable_patterns[idiom] = patterns

            if self.verbose:
                print(f"   ✅ {idiom} → {display_form} ({len(patterns)}개 패턴)")

        print(f"✅ 분리형 패턴 생성 완료: {len(self.user_separable_idioms)}개 숙어")

    def _generate_detection_patterns(self, verb: str, particle: str) -> List[Dict]:
        """개별 분리형 숙어의 감지 패턴들 생성"""
        patterns = []

        # 1. 연속형 (기본형)
        patterns.append(
            {
                "pattern": rf"\b{re.escape(verb)}\s+{re.escape(particle)}\b",
                "type": "continuous",
                "description": "연속형",
                "is_separated": False,
                "priority": 1,
            }
        )

        # 2. 분리형 - 일반 명사/명사구
        patterns.append(
            {
                "pattern": rf"\b{re.escape(verb)}\s+(?:the\s+|a\s+|an\s+|this\s+|that\s+|his\s+|her\s+|my\s+|your\s+|our\s+|their\s+)?(?:\w+\s+)*\w+\s+{re.escape(particle)}\b",
                "type": "separated_noun",
                "description": "분리형(명사)",
                "is_separated": True,
                "priority": 2,
            }
        )

        # 3. 분리형 - 대명사 (반드시 분리)
        patterns.append(
            {
                "pattern": rf"\b{re.escape(verb)}\s+(?:it|him|her|them|this|that|these|those)\s+{re.escape(particle)}\b",
                "type": "separated_pronoun",
                "description": "분리형(대명사)",
                "is_separated": True,
                "priority": 3,
            }
        )

        # 4. 분리형 - 긴 명사구
        patterns.append(
            {
                "pattern": rf"\b{re.escape(verb)}\s+(?:the\s+)?(?:\w+\s+){{2,}}\w+\s+{re.escape(particle)}\b",
                "type": "separated_long_noun",
                "description": "분리형(긴명사구)",
                "is_separated": True,
                "priority": 2,
            }
        )

        return patterns

    def detect_separable_idioms_in_text(self, text: str) -> List[Dict]:
        """텍스트에서 사용자 DB 분리형 숙어 감지"""
        if not self.user_separable_idioms:
            return []

        results = []
        found_positions = set()

        if self.verbose:
            print(f"🔍 분리형 숙어 감지 중: {len(self.user_separable_idioms)}개 대상")

        # 우선순위순으로 정렬 (분리형을 먼저 찾아서 연속형과 중복 방지)
        for idiom, info in self.user_separable_idioms.items():
            patterns = self.separable_patterns.get(idiom, [])

            # 우선순위순으로 패턴 정렬 (분리형 먼저)
            sorted_patterns = sorted(
                patterns, key=lambda x: x["priority"], reverse=True
            )

            for pattern_info in sorted_patterns:
                pattern = pattern_info["pattern"]

                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    start, end = match.span()

                    # 중복 위치 확인
                    if any(abs(start - pos[0]) <= 5 for pos in found_positions):
                        continue

                    matched_text = match.group().strip()

                    # 결과 생성
                    result = {
                        "original": matched_text,
                        "base_form": idiom,
                        "display_form": info["display_form"],
                        "pattern_type": pattern_info["type"],
                        "description": pattern_info["description"],
                        "is_separated": pattern_info["is_separated"],
                        "confidence": info["confidence"],
                        "start": start,
                        "end": end,
                        "separable_info": {
                            "verb": info["verb"],
                            "particle": info["particle"],
                            "detection_pattern": pattern,
                            "gpt_reason": info.get("reason", ""),
                            "examples": info.get("examples", []),
                        },
                    }

                    results.append(result)
                    found_positions.add((start, end))

                    if self.verbose:
                        sep_mark = "🔧" if result["is_separated"] else "📝"
                        print(
                            f"      {sep_mark} 발견: '{matched_text}' → {info['display_form']} ({pattern_info['description']})"
                        )

                    # 같은 숙어의 다른 패턴은 스킵 (첫 번째 매치만)
                    break

        return results

    def integrate_with_extractor(self, extractor_instance):
        """기존 AdvancedVocabExtractor와 통합"""
        print(f"🔗 기존 extractor와 분리형 숙어 감지기 통합...")

        # 기존 사용자 숙어들 분석
        if hasattr(extractor_instance, "user_idioms"):
            separable_analysis = self.analyze_user_idioms_with_gpt(
                extractor_instance.user_idioms
            )
            self.build_separable_patterns(separable_analysis)

            # extractor에 분리형 정보 추가
            extractor_instance.separable_detector = self
            extractor_instance.user_separable_idioms = self.user_separable_idioms

            print(
                f"✅ 통합 완료: {len(self.user_separable_idioms)}개 분리형 숙어 활성화"
            )
        else:
            print("⚠️ extractor에 user_idioms가 없습니다")

    def save_separable_analysis(self, output_file: str = "separable_analysis.json"):
        """분리형 분석 결과 저장"""
        analysis_data = {
            "separable_idioms": self.user_separable_idioms,
            "total_count": len(self.user_separable_idioms),
            "gpt_calls": self.gpt_calls,
            "cache": self.separable_cache,
        }

        if save_json_safe(analysis_data, output_file):
            print(f"✅ 분리형 분석 결과 저장: {output_file}")
        else:
            print(f"❌ 저장 실패: {output_file}")

    def load_separable_analysis(self, input_file: str = "separable_analysis.json"):
        """기존 분리형 분석 결과 로드"""
        analysis_data = load_json_safe(input_file)

        if not analysis_data:
            print(f"📂 {input_file} 파일이 없습니다. 새로 분석합니다.")
            return False

        try:
            self.user_separable_idioms = analysis_data.get("separable_idioms", {})
            self.separable_cache = analysis_data.get("cache", {})

            # 패턴 재생성
            for idiom, info in self.user_separable_idioms.items():
                verb = info.get("verb", "")
                particle = info.get("particle", "")
                if verb and particle:
                    patterns = self._generate_detection_patterns(verb, particle)
                    self.separable_patterns[idiom] = patterns

            print(f"✅ 분리형 분석 결과 로드: {len(self.user_separable_idioms)}개")
            return True

        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return False


class AdvancedIdiomChecker:
    """고급 숙어 검증기 - 분리형과 문법 패턴 구분"""

    def __init__(self, nlp_model):
        self.nlp = nlp_model

        # 연속형 가능 구동사 (붙여서 써도 OK)
        self.optional_separable = {
            "pick up",
            "turn on",
            "turn off",
            "look up",
            "put on",
            "take off",
            "bring up",
            "call up",
            "give up",
            "set up",
            "clean up",
            "fill up",
            "write down",
            "sit down",
            "stand up",
            "wake up",
            "get up",
        }

        # 반드시 분리되어야 하는 구동사 (목적어 필수)
        self.mandatory_separable = {
            "pick": ["up", "out", "off"],
            "turn": ["down", "up"],
            "put": ["away", "back"],
            "take": ["apart", "down"],
            "figure": ["out"],
            "work": ["out"],
            "point": ["out"],
            "carry": ["out"],
            "bring": ["about"],
        }

        # 문법 패턴 숙어들 (특정 품사 필수)
        self.grammar_patterns = {
            # V-ing 패턴
            r"\bspend\s+(?:time|money|hours?|days?|years?)\s+(\w+ing)\b": "spend time V-ing",
            r"\bis\s+worth\s+(\w+ing)\b": "be worth V-ing",
            r"\bkeep\s+(?:on\s+)?(\w+ing)\b": "keep V-ing",
            r"\bavoid\s+(\w+ing)\b": "avoid V-ing",
            r"\benjoy\s+(\w+ing)\b": "enjoy V-ing",
            r"\bfinish\s+(\w+ing)\b": "finish V-ing",
            # N + V-ing 패턴
            r"\bprevent\s+(\w+)\s+from\s+(\w+ing)\b": "prevent N from V-ing",
            r"\bstop\s+(\w+)\s+from\s+(\w+ing)\b": "stop N from V-ing",
            # 기타 패턴
            r"\bit\s+takes\s+(\w+)\s+to\s+(\w+)": "it takes N to V",
            r"\bthere\s+is\s+no\s+point\s+in\s+(\w+ing)\b": "there is no point in V-ing",
        }

        # 알려진 일반 숙어 패턴들
        self.known_phrasal_patterns = {
            r"\bas\s+\w+\s+as\b",
            r"\bin\s+order\s+to\b",
            r"\bas\s+a\s+result\b",
            r"\bon\s+the\s+other\s+hand\b",
            r"\bfor\s+instance\b",
            r"\bin\s+spite\s+of\b",
            r"\bbecause\s+of\b",
            r"\binstead\s+of\b",
            r"\baccording\s+to\b",
        }

    def analyze_phrasal_verb_pattern(self, text, verb_token, particle_token):
        """구동사 패턴 분석 - 연속형 vs 분리형 vs 문법패턴"""
        verb = verb_token.lemma_.lower()
        particle = particle_token.text.lower()
        base_phrasal = f"{verb} {particle}"

        # 1. 연속형 가능한 구동사인지 확인
        if base_phrasal in self.optional_separable:
            return {
                "pattern_type": "optional_separable",
                "base_form": base_phrasal,
                "display_form": base_phrasal,  # 그냥 연속형으로 표시
                "is_separated": False,
            }

        # 2. 반드시 분리되어야 하는 구동사인지 확인
        if (
            verb in self.mandatory_separable
            and particle in self.mandatory_separable[verb]
        ):
            return {
                "pattern_type": "mandatory_separable",
                "base_form": base_phrasal,
                "display_form": f"{verb} ~ {particle}",  # ~ 로 표시
                "is_separated": True,
            }

        # 3. 일반 구동사 (실제 분리되었는지 확인)
        verb_idx = verb_token.i
        particle_idx = particle_token.i

        # 동사와 입자 사이에 다른 토큰이 있는지 확인
        if abs(verb_idx - particle_idx) > 1:
            # 실제로 분리되어 있음
            return {
                "pattern_type": "actually_separated",
                "base_form": base_phrasal,
                "display_form": f"{verb} ~ {particle}",
                "is_separated": True,
            }
        else:
            # 연속으로 붙어있음
            return {
                "pattern_type": "continuous",
                "base_form": base_phrasal,
                "display_form": base_phrasal,
                "is_separated": False,
            }

    def analyze_grammar_pattern(self, text):
        """문법 패턴 분석"""
        results = []

        for pattern_regex, pattern_name in self.grammar_patterns.items():
            matches = re.finditer(pattern_regex, text, re.IGNORECASE)
            for match in matches:
                start, end = match.span()
                original_text = match.group()

                # 매칭된 그룹들 분석
                groups = match.groups()

                # V-ing 패턴 검증
                if "V-ing" in pattern_name and groups:
                    # 마지막 그룹이 실제로 동명사인지 확인
                    last_word = groups[-1]
                    if last_word.endswith("ing"):
                        results.append(
                            {
                                "original": original_text,
                                "pattern_type": "grammar_pattern",
                                "display_form": pattern_name,
                                "base_form": pattern_name,
                                "start": start,
                                "end": end,
                                "is_separated": False,
                            }
                        )

                # 기타 패턴
                else:
                    results.append(
                        {
                            "original": original_text,
                            "pattern_type": "grammar_pattern",
                            "display_form": pattern_name,
                            "base_form": pattern_name,
                            "start": start,
                            "end": end,
                            "is_separated": False,
                        }
                    )

        return results
