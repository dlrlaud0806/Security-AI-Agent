from typing import Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langsmith import traceable
from ..config.settings import settings
from ..utils.langsmith_config import LangSmithTracker

QuestionType = Literal["faq", "sap_automation", "data_request"]

class ClassificationResult(BaseModel):
    question_type: QuestionType = Field(description="질문의 분류 타입")
    confidence: float = Field(description="분류 신뢰도 (0.0-1.0)", ge=0.0, le=1.0)
    reasoning: str = Field(description="분류 근거")

class QuestionClassificationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=ClassificationResult)
        self.tracker = LangSmithTracker("question_classifier")
        
        self.system_prompt = """당신은 사용자 질문을 다음 3가지 카테고리로 분류하는 전문가입니다:

1. **faq**: 일반적인 도움말, 사용법 문의, 기본적인 질문
   - 예시: "어떻게 사용하나요?", "도움말이 필요해요", "이게 뭔가요?"
   
2. **sap_automation**: SAP 시스템 자동화, 업무 프로세스, 워크플로우 관련
   - 예시: "SAP에서 주문 생성을 자동화하고 싶어요", "id 락해제 해주세요", "비밀번호 초기화 해주세요", "SAP GUI에서 특정 프로세스 자동화 해주세요"
   
3. **data_request**: 데이터 조회, 검색, 리포트, 통계 요청
   - 예시: "작년 매출 데이터를 보여주세요", "사용자 정보를 검색하고 싶어요", "특정 권한 보유한 사용자 알려주세요"

분류할 때 다음을 고려하세요:
- 질문의 핵심 의도와 목적
- 사용된 키워드와 맥락
- 요청하는 작업의 성격

불확실한 경우 faq로 분류하고 신뢰도를 낮게 설정하세요.

{format_instructions}"""

    @traceable(name="classify_question")
    def classify_question(self, question: str) -> ClassificationResult:
        try:
            messages = [
                SystemMessage(content=self.system_prompt.format(
                    format_instructions=self.parser.get_format_instructions()
                )),
                HumanMessage(content=f"다음 질문을 분류해주세요: {question}")
            ]
            
            response = self.llm.invoke(messages)
            result = self.parser.parse(response.content)
            return result
            
        except Exception as e:
            return ClassificationResult(
                question_type="faq",
                confidence=0.1,
                reasoning=f"분류 중 오류 발생: {str(e)}, 기본값으로 faq 반환"
            )
    
    @traceable(name="classify_with_fallback")
    def classify_with_fallback(self, question: str) -> Dict[str, Any]:
        result = self.classify_question(question)
        
        if result.confidence < 0.3:
            fallback_result = self._fallback_classification(question)
            return {
                "question_type": fallback_result,
                "confidence": 0.5,
                "reasoning": f"LLM 분류 신뢰도 낮음({result.confidence:.2f}), 키워드 기반 폴백 사용",
                "original_classification": {
                    "type": result.question_type,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning
                }
            }
        
        return {
            "question_type": result.question_type,
            "confidence": result.confidence,
            "reasoning": result.reasoning
        }
    
    def _fallback_classification(self, question: str) -> QuestionType:
        faq_keywords = ["도움말", "help", "사용법", "how to", "what is", "무엇", "어떻게", "에러", "오류"]
        sap_keywords = ["sap", "자동화", "gui", "락해제", "process", "업무"]
        data_keywords = ["데이터", "data", "정보", "조회", "검색", "리포트", "report", "통계"]
        
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in sap_keywords):
            return "sap_automation"
        elif any(keyword in question_lower for keyword in data_keywords):
            return "data_request"
        elif any(keyword in question_lower for keyword in faq_keywords):
            return "faq"
        else:
            return "faq"