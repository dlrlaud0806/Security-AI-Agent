from typing import Dict, Any, List, Literal
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langsmith import traceable
from ..config.settings import settings
from ..utils.langsmith_config import LangSmithTracker

SafetyLevel = Literal["safe", "warning", "blocked"]

class SafetyAssessment(BaseModel):
    safety_level: SafetyLevel = Field(description="안전성 평가 등급")
    confidence: float = Field(description="평가 신뢰도 (0.0-1.0)", ge=0.0, le=1.0)
    risk_categories: List[str] = Field(description="감지된 위험 카테고리 목록", default=[])
    reasoning: str = Field(description="안전성 평가 근거")
    recommended_action: str = Field(description="권장 조치")

class OutputSafetyAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=SafetyAssessment)
        self.tracker = LangSmithTracker("output_safety_agent")
        
        self.system_prompt = """당신은 사용자 요청의 안전성을 평가하는 보안 전문가입니다.

다음 기준으로 요청을 평가하세요:

**위험 카테고리:**
1. **personal_info**: 개인정보 (주민번호, 여권번호, 신용카드, 전화번호 등)
2. **credentials**: 인증정보 (비밀번호, API 키, 토큰, 인증서 등)
3. **financial**: 금융정보 (계좌번호, 카드번호, 거래내역 등)
4. **health**: 의료정보 (병력, 진료기록, 의료데이터 등)
5. **corporate**: 기업기밀 (영업비밀, 내부정보, 전략정보 등)
6. **system**: 시스템정보 (서버정보, DB 스키마, 네트워크 구성 등)

**안전성 등급:**
- **safe**: 안전한 요청, 제한 없음
- **warning**: 주의가 필요하지만 제한적 처리 가능
- **blocked**: 위험한 요청, 처리 차단

**평가 기준:**
- 요청의 의도와 맥락을 종합적으로 고려
- 직접적인 민감정보 요청은 blocked
- 간접적이거나 학습 목적은 warning
- 일반적인 업무 요청은 safe

**예시:**
- "고객의 주민번호를 알려주세요" → blocked
- "주민번호 검증 방법을 알려주세요" → warning  
- "SAP에서 주문 조회하는 방법은?" → safe

{format_instructions}"""

    @traceable(name="assess_safety")
    def assess_safety(self, user_request: str) -> SafetyAssessment:
        try:
            messages = [
                SystemMessage(content=self.system_prompt.format(
                    format_instructions=self.parser.get_format_instructions()
                )),
                HumanMessage(content=f"다음 사용자 요청의 안전성을 평가해주세요: {user_request}")
            ]
            
            response = self.llm.invoke(messages)
            result = self.parser.parse(response.content)
            return result
            
        except Exception as e:
            return SafetyAssessment(
                safety_level="blocked",
                confidence=1.0,
                risk_categories=["system_error"],
                reasoning=f"안전성 평가 중 오류 발생: {str(e)}",
                recommended_action="요청을 차단하고 시스템 관리자에게 문의"
            )
    
    @traceable(name="assess_with_fallback")
    def assess_with_fallback(self, user_request: str) -> Dict[str, Any]:
        result = self.assess_safety(user_request)
        
        if result.confidence < 0.3:
            fallback_result = self._fallback_assessment(user_request)
            return {
                "safety_level": fallback_result["safety_level"],
                "confidence": 0.5,
                "risk_categories": fallback_result["risk_categories"],
                "reasoning": f"LLM 평가 신뢰도 낮음({result.confidence:.2f}), 키워드 기반 폴백 사용",
                "recommended_action": fallback_result["recommended_action"],
                "original_assessment": {
                    "safety_level": result.safety_level,
                    "confidence": result.confidence,
                    "risk_categories": result.risk_categories,
                    "reasoning": result.reasoning,
                    "recommended_action": result.recommended_action
                }
            }
        
        return {
            "safety_level": result.safety_level,
            "confidence": result.confidence,
            "risk_categories": result.risk_categories,
            "reasoning": result.reasoning,
            "recommended_action": result.recommended_action
        }
    
    def _fallback_assessment(self, user_request: str) -> Dict[str, Any]:
        request_lower = user_request.lower()
        
        high_risk_patterns = [
            "password", "비밀번호", "secret", "비밀", "token", "토큰",
            "주민번호", "ssn", "여권번호", "passport", "신용카드", "카드번호",
            "계좌번호", "account", "api_key", "private_key"
        ]
        
        medium_risk_patterns = [
            "credential", "인증", "private", "개인정보", "의료", "병력",
            "급여", "salary", "내부", "기밀", "confidential"
        ]
        
        detected_risks = []
        
        if any(pattern in request_lower for pattern in high_risk_patterns):
            detected_risks.extend(["credentials", "personal_info"])
            return {
                "safety_level": "blocked",
                "risk_categories": detected_risks,
                "recommended_action": "요청 차단 - 고위험 정보 포함"
            }
        elif any(pattern in request_lower for pattern in medium_risk_patterns):
            detected_risks.extend(["personal_info"])
            return {
                "safety_level": "warning", 
                "risk_categories": detected_risks,
                "recommended_action": "제한적 처리 - 주의 필요"
            }
        else:
            return {
                "safety_level": "safe",
                "risk_categories": [],
                "recommended_action": "정상 처리"
            }