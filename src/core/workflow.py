from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage
from langsmith import traceable
from ..agents.security_agent import PromptInjectionDetector
from ..agents.question_classifier import QuestionClassificationAgent
from ..agents.output_safety_agent import OutputSafetyAgent
from ..agents.final_safety_agent import FinalSafetyAgent
from ..agents.sap_automation_agent import SAPAutomationAgent
from ..utils.langsmith_config import LangSmithTracker, setup_langsmith
from .chatbot import Chatbot

class ChatbotState:
    def __init__(self):
        self.user_input: str = ""
        self.response: str = ""
        self.should_block: bool = False
        self._metadata: Dict[str, Any] = {}
    
    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)
    
    def get_all_metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()
    
    def clear_metadata_except(self, keep_keys: list) -> None:
        self._metadata = {k: v for k, v in self._metadata.items() if k in keep_keys}

class SecureChatbotWorkflow:
    def __init__(self, system_prompt: str = "You are a helpful AI assistant."):
        setup_langsmith()
        
        self.security_agent = PromptInjectionDetector()
        self.question_classifier = QuestionClassificationAgent()
        self.output_safety_agent = OutputSafetyAgent()
        self.final_safety_agent = FinalSafetyAgent()
        self.sap_automation_agent = SAPAutomationAgent()
        self.chatbot = Chatbot(system_prompt)
        self.tracker = LangSmithTracker("secure_chatbot_workflow")
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(dict)
        
        workflow.add_node("security_check", self._security_check_node)
        workflow.add_node("process_message", self._process_message_node)
        workflow.add_node("classify_question", self._classify_question_node)
        workflow.add_node("output_safety_check", self._output_safety_check_node)
        workflow.add_node("sap_automation", self._sap_automation_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("final_safety_check", self._final_safety_check_node)
        
        workflow.set_entry_point("security_check")
        
        workflow.add_conditional_edges(
            "security_check",
            self._should_block_message,
            {
                "block": END,
                "continue": "process_message"
            }
        )
        
        workflow.add_edge("process_message", "classify_question")
        
        workflow.add_conditional_edges(
            "classify_question",
            self._route_by_question_type,
            {
                "faq": "generate_response",
                "sap_automation": "sap_automation",
                "data_request": "output_safety_check"
            }
        )
        
        workflow.add_edge("output_safety_check", "generate_response")
        workflow.add_edge("sap_automation", "generate_response")
        workflow.add_edge("generate_response", "final_safety_check")
        
        workflow.add_conditional_edges(
            "final_safety_check",
            self._should_block_final_response,
            {
                "block": END,
                "allow": END
            }
        )
        
        return workflow.compile()
    
    def _check_input_length(self, user_input: str) -> Dict[str, Any]:
        """입력 길이 검사를 수행하는 단순 함수"""
        max_length = 300
        input_length = len(user_input)
        
        if input_length > max_length:
            return {
                "response": f"입력이 너무 깁니다. 최대 {max_length}자까지 입력 가능합니다. (현재: {input_length}자)\n간단하고 명확하게 질문해 주세요.",
                "blocked": True,
                "length_check": {
                    "blocked": True,
                    "length": input_length,
                    "max_length": max_length,
                    "reason": "입력 길이 제한 초과"
                },
                "security_check": {},
                "classification": {},
                "safety_assessment": {},
                "final_safety_assessment": {},
                "final_response_blocked": False,
                "sap_automation_result": {}
            }
        
        return {
            "blocked": False,
            "length_check": {
                "blocked": False,
                "length": input_length,
                "max_length": max_length,
                "reason": "입력 길이 적정"
            }
        }
    
    @traceable(name="security_check_node")
    def _security_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        security_result = self.security_agent.detect_injection(user_input)
        
        state["should_block"] = security_result["is_malicious"]
        if security_result["is_malicious"]:
            state["response"] = "I cannot process that request as it appears to contain potentially harmful instructions."
            state["_security_check"] = security_result
        else:
            state["_security_check"] = {"is_malicious": False, "confidence": security_result.get("confidence", 0)}
        
        return state
    
    def _should_block_message(self, state: Dict[str, Any]) -> str:
        return "block" if state.get("should_block", False) else "continue"
    
    @traceable(name="process_message_node")
    def _process_message_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        sanitized_input = self.security_agent.sanitize_input(user_input)
        state["sanitized_input"] = sanitized_input
        return state
    
    @traceable(name="classify_question_node")
    def _classify_question_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_input = state.get("sanitized_input", "")
        
        classification_result = self.question_classifier.classify_with_fallback(sanitized_input)
        
        state["question_type"] = classification_result["question_type"]
        state["_classification"] = {
            "confidence": classification_result["confidence"],
            "reasoning": classification_result["reasoning"]
        }
        
        if "original_classification" in classification_result:
            state["_classification"]["original_classification"] = classification_result["original_classification"]
        
        return state
    
    def _route_by_question_type(self, state: Dict[str, Any]) -> str:
        return state.get("question_type", "faq")
    
    @traceable(name="output_safety_check_node")
    def _output_safety_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_input = state.get("sanitized_input", "")
        
        safety_result = self.output_safety_agent.assess_with_fallback(sanitized_input)
        
        state["output_safety_approved"] = safety_result["safety_level"] == "safe"
        state["_safety_assessment"] = safety_result
        
        if safety_result["safety_level"] != "safe":
            state["safety_warning"] = f"{'보안 위험' if safety_result['safety_level'] == 'blocked' else '주의 필요'}: {safety_result['recommended_action']}"
        
        return state

    @traceable(name="sap_automation_node")
    def _sap_automation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_input = state.get("sanitized_input", "")
        
        try:
            sap_result = self.sap_automation_agent.process_user_request(sanitized_input)
            
            if sap_result.success:
                state["response"] = f"✅ {sap_result.message}\n\n{sap_result.details}"
            else:
                state["response"] = f"❌ {sap_result.message}\n\n{sap_result.details}"
                
            state["_sap_automation_result"] = {
                "success": sap_result.success,
                "message": sap_result.message,
                "details": sap_result.details
            }
            
        except Exception as e:
            state["response"] = f"❌ SAP 자동화 처리 중 오류가 발생했습니다: {str(e)}"
            state["_sap_automation_result"] = {
                "success": False,
                "message": "처리 오류",
                "details": str(e)
            }
        
        return state

    def _cleanup_intermediate_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        keys_to_remove = ["sanitized_input", "safety_warning"]
        for key in keys_to_remove:
            state.pop(key, None)
        return state

    @traceable(name="generate_response_node")
    def _generate_response_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_input = state.get("sanitized_input", "")
        question_type = state.get("question_type", "faq")
        output_safety_approved = state.get("output_safety_approved", True)
        
        # SAP 자동화의 경우 이미 응답이 생성되었으므로 그대로 반환
        if question_type == "sap_automation" and "response" in state:
            return state
        
        safety_assessment = state.get("_safety_assessment", {})
        
        if not output_safety_approved:
            if safety_assessment.get("safety_level") == "blocked":
                state["response"] = "죄송합니다. 보안상 위험한 요청으로 판단되어 처리할 수 없습니다."
            else:
                state["response"] = "죄송합니다. 민감한 정보와 관련된 요청은 처리할 수 없습니다."
            return state
        
        response = self.chatbot.chat(sanitized_input)
        
        if state.get("safety_warning"):
            response += f"\n\n⚠️ {state['safety_warning']}"
            
        state["response"] = response
        
        return self._cleanup_intermediate_data(state)

    @traceable(name="final_safety_check_node")
    def _final_safety_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = state.get("response", "")
        
        if not response:
            return state
        
        final_safety_result = self.final_safety_agent.assess_with_fallback(response)
        
        state["_final_safety_assessment"] = final_safety_result
        state["final_response_blocked"] = final_safety_result["safety_level"] == "blocked"
        
        if final_safety_result["safety_level"] == "blocked":
            state["response"] = "죄송합니다. 생성된 응답이 보안 정책에 위배되어 제공할 수 없습니다."
            state["final_safety_warning"] = f"최종 보안 차단: {final_safety_result['recommended_action']}"
        elif final_safety_result["safety_level"] == "warning":
            state["final_safety_warning"] = f"최종 보안 주의: {final_safety_result['recommended_action']}"
            state["response"] += f"\n\n⚠️ {state['final_safety_warning']}"
        
        return state

    def _should_block_final_response(self, state: Dict[str, Any]) -> str:
        return "block" if state.get("final_response_blocked", False) else "allow"
    
    @traceable(name="process_message")
    def process_message(self, user_input: str) -> Dict[str, Any]:
        from datetime import datetime
        
        # 1. 최우선 길이 체크 (가장 빠른 차단, 보안 체크 이전)
        length_check_result = self._check_input_length(user_input)
        if length_check_result["blocked"]:
            return length_check_result
        
        # 2. LangGraph 보안 워크플로우 실행
        initial_state = {
            "user_input": user_input,
            "security_check": {},
            "response": "",
            "should_block": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            
            classification = result.get("_classification", {})
            return {
                "response": result["response"],
                "length_check": length_check_result["length_check"],  # 길이 체크 결과 포함
                "security_check": result.get("_security_check", {}),
                "blocked": result["should_block"],
                "classification": {
                    "question_type": result.get("question_type"),
                    "confidence": classification.get("confidence"),
                    "reasoning": classification.get("reasoning"),
                    "original_classification": classification.get("original_classification")
                },
                "safety_assessment": result.get("_safety_assessment", {}),
                "final_safety_assessment": result.get("_final_safety_assessment", {}),
                "final_response_blocked": result.get("final_response_blocked", False),
                "sap_automation_result": result.get("_sap_automation_result", {})
            }
        except Exception as e:
            raise
    
    def clear_history(self):
        self.chatbot.clear_history()
    
    def get_conversation_history(self):
        return self.chatbot.get_conversation_history()