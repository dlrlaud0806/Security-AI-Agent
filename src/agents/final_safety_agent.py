from typing import Dict, Any, List, Literal
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langsmith import traceable
import re
from ..config.settings import settings
from ..utils.langsmith_config import LangSmithTracker

ResponseSafetyLevel = Literal["safe", "warning", "blocked"]

class ResponseSafetyAssessment(BaseModel):
    safety_level: ResponseSafetyLevel = Field(description="응답 안전성 등급")
    confidence: float = Field(description="평가 신뢰도 (0.0-1.0)", ge=0.0, le=1.0)
    detected_issues: List[str] = Field(description="감지된 보안 이슈 목록", default=[])
    reasoning: str = Field(description="안전성 평가 근거")
    recommended_action: str = Field(description="권장 조치")

class FinalSafetyAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=ResponseSafetyAssessment)
        self.tracker = LangSmithTracker("final_safety_agent")
        
        self.system_prompt = """당신은 AI가 생성한 응답의 최종 안전성을 검증하는 보안 전문가입니다.

**검사 대상: AI 생성 응답 콘텐츠**

**주요 검사 항목:**

1. **실제 민감정보 노출**
   - 실제 개인정보: 주민번호, 전화번호, 주소, 이메일 등
   - 실제 인증정보: 비밀번호, API 키, 토큰 등
   - 실제 금융정보: 계좌번호, 카드번호 등
   - 실제 기업기밀: 내부 문서, 코드, 데이터베이스 스키마 등

2. **위험한 코드/명령어**
   - 시스템 손상 코드: rm -rf /, del /f /s /q, DROP TABLE 등
   - 네트워크 공격 코드: 포트 스캐닝, SQL 인젝션, XSS 등
   - 악성 스크립트: 백도어, 키로거, 랜섬웨어 등

3. **잘못된 정보 생성**
   - LLM이 학습 데이터에서 추출한 가짜 개인정보
   - 존재하지 않는 시스템 정보를 실제처럼 제시
   - 잘못된 보안 가이드라인 제공

4. **규정 위반 콘텐츠**
   - 불법 활동 조장
   - 해킹/크래킹 상세 가이드
   - 규제 위반 정보 제공

**안전성 등급:**
- **safe**: 안전한 응답, 문제없음
- **warning**: 주의가 필요하지만 경고와 함께 제공 가능
- **blocked**: 위험한 응답, 차단 필요

**평가 원칙:**
- 응답에 실제 민감정보가 포함되었는지 확인
- 코드나 명령어가 실제 위험을 초래할 수 있는지 판단
- 일반적인 설명이나 교육적 내용은 safe로 분류
- 구체적이고 실행 가능한 위험 정보는 blocked로 분류

{format_instructions}"""

        # 위험한 패턴들 (정규표현식)
        self.dangerous_patterns = {
            'system_commands': [
                r'rm\s+-rf\s+/',
                r'del\s+/[fs]\s+/[sq]\s+',
                r'DROP\s+TABLE',
                r'shutdown\s+/[srf]',
                r'format\s+c:',
            ],
            'sensitive_info': [
                r'\d{3}-\d{2}-\d{4}',  # SSN pattern
                r'\d{6}-\d{7}',        # Korean ID pattern
                r'\d{4}-\d{4}-\d{4}-\d{4}',  # Credit card pattern
                r'[A-Za-z0-9]{40,}',   # Long tokens/keys
            ],
            'network_attacks': [
                r'nmap\s+-',
                r'sqlmap\s+',
                r'<script[^>]*>.*</script>',
                r'union\s+select.*from',
            ]
        }

    def _check_dangerous_patterns(self, response: str) -> List[str]:
        """위험한 패턴 검사"""
        detected_issues = []
        
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    detected_issues.append(f"{category}: {pattern}")
        
        return detected_issues

    @traceable(name="assess_response_safety")
    def assess_response_safety(self, ai_response: str) -> ResponseSafetyAssessment:
        try:
            # 먼저 패턴 기반 검사 수행
            pattern_issues = self._check_dangerous_patterns(ai_response)
            
            messages = [
                SystemMessage(content=self.system_prompt.format(
                    format_instructions=self.parser.get_format_instructions()
                )),
                HumanMessage(content=f"다음 AI 생성 응답의 안전성을 평가해주세요:\n\n{ai_response}")
            ]
            
            response = self.llm.invoke(messages)
            result = self.parser.parse(response.content)
            
            # 패턴 검사 결과를 LLM 결과와 결합
            if pattern_issues:
                result.detected_issues.extend(pattern_issues)
                if result.safety_level == "safe":
                    result.safety_level = "blocked"
                    result.reasoning += f" (추가: 위험한 패턴 감지 - {', '.join(pattern_issues)})"
            
            return result
            
        except Exception as e:
            return ResponseSafetyAssessment(
                safety_level="blocked",
                confidence=1.0,
                detected_issues=["system_error"],
                reasoning=f"최종 안전성 평가 중 오류 발생: {str(e)}",
                recommended_action="응답을 차단하고 시스템 관리자에게 문의"
            )

    @traceable(name="assess_with_fallback")
    def assess_with_fallback(self, ai_response: str) -> Dict[str, Any]:
        result = self.assess_response_safety(ai_response)
        
        if result.confidence < 0.3:
            fallback_result = self._fallback_assessment(ai_response)
            return {
                "safety_level": fallback_result["safety_level"],
                "confidence": 0.5,
                "detected_issues": fallback_result["detected_issues"],
                "reasoning": f"LLM 평가 신뢰도 낮음({result.confidence:.2f}), 패턴 기반 폴백 사용",
                "recommended_action": fallback_result["recommended_action"],
                "original_assessment": {
                    "safety_level": result.safety_level,
                    "confidence": result.confidence,
                    "detected_issues": result.detected_issues,
                    "reasoning": result.reasoning,
                    "recommended_action": result.recommended_action
                }
            }
        
        return {
            "safety_level": result.safety_level,
            "confidence": result.confidence,
            "detected_issues": result.detected_issues,
            "reasoning": result.reasoning,
            "recommended_action": result.recommended_action
        }

    def _fallback_assessment(self, ai_response: str) -> Dict[str, Any]:
        """패턴 기반 폴백 평가"""
        response_lower = ai_response.lower()
        detected_issues = []
        
        # 고위험 패턴
        high_risk_keywords = [
            "rm -rf", "delete", "drop table", "truncate", "shutdown",
            "password:", "api_key:", "secret_key:", "token:",
            "주민번호", "여권번호", "계좌번호", "비밀번호"
        ]
        
        # 중위험 패턴  
        medium_risk_keywords = [
            "private", "confidential", "internal", "credential",
            "hack", "crack", "exploit", "vulnerability"
        ]
        
        for keyword in high_risk_keywords:
            if keyword in response_lower:
                detected_issues.append(f"high_risk_keyword: {keyword}")
        
        for keyword in medium_risk_keywords:
            if keyword in response_lower:
                detected_issues.append(f"medium_risk_keyword: {keyword}")
        
        if any("high_risk" in issue for issue in detected_issues):
            return {
                "safety_level": "blocked",
                "detected_issues": detected_issues,
                "recommended_action": "응답 차단 - 고위험 콘텐츠 감지"
            }
        elif any("medium_risk" in issue for issue in detected_issues):
            return {
                "safety_level": "warning",
                "detected_issues": detected_issues,
                "recommended_action": "경고 메시지와 함께 제한적 제공"
            }
        else:
            return {
                "safety_level": "safe",
                "detected_issues": [],
                "recommended_action": "정상 응답 제공"
            }