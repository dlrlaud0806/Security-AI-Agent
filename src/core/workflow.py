from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage
from langsmith import traceable
from ..agents.security_agent import PromptInjectionDetector
from ..agents.question_classifier import QuestionClassificationAgent
from ..agents.output_safety_agent import OutputSafetyAgent
from ..utils.langsmith_config import LangSmithTracker, setup_langsmith
from .chatbot import Chatbot

class ChatbotState:
    def __init__(self):
        self.user_input: str = ""
        self.security_check: Dict[str, Any] = {}
        self.response: str = ""
        self.should_block: bool = False

class SecureChatbotWorkflow:
    def __init__(self, system_prompt: str = "You are a helpful AI assistant."):
        setup_langsmith()
        
        self.security_agent = PromptInjectionDetector()
        self.question_classifier = QuestionClassificationAgent()
        self.output_safety_agent = OutputSafetyAgent()
        self.chatbot = Chatbot(system_prompt)
        self.tracker = LangSmithTracker("secure_chatbot_workflow")
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(dict)
        
        workflow.add_node("security_check", self._security_check_node)
        workflow.add_node("process_message", self._process_message_node)
        workflow.add_node("classify_question", self._classify_question_node)
        workflow.add_node("output_safety_check", self._output_safety_check_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
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
                "sap_automation": "generate_response",
                "data_request": "output_safety_check"
            }
        )
        
        workflow.add_edge("output_safety_check", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    @traceable(name="security_check_node")
    def _security_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        security_result = self.security_agent.detect_injection(user_input)
        
        state["security_check"] = security_result
        state["should_block"] = security_result["is_malicious"]
        
        if security_result["is_malicious"]:
            state["response"] = "I cannot process that request as it appears to contain potentially harmful instructions."
        
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
        state["classification_confidence"] = classification_result["confidence"]
        state["classification_reasoning"] = classification_result["reasoning"]
        
        if "original_classification" in classification_result:
            state["original_classification"] = classification_result["original_classification"]
        
        return state
    
    def _route_by_question_type(self, state: Dict[str, Any]) -> str:
        return state.get("question_type", "faq")
    
    @traceable(name="output_safety_check_node")
    def _output_safety_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_input = state.get("sanitized_input", "")
        
        safety_result = self.output_safety_agent.assess_with_fallback(sanitized_input)
        
        state["safety_assessment"] = safety_result
        state["output_safety_approved"] = safety_result["safety_level"] == "safe"
        
        if safety_result["safety_level"] == "blocked":
            state["safety_warning"] = f"보안 위험: {safety_result['recommended_action']}"
        elif safety_result["safety_level"] == "warning":
            state["safety_warning"] = f"주의 필요: {safety_result['recommended_action']}"
        
        return state

    @traceable(name="generate_response_node")
    def _generate_response_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_input = state.get("sanitized_input", "")
        question_type = state.get("question_type", "faq")
        output_safety_approved = state.get("output_safety_approved", True)
        
        safety_assessment = state.get("safety_assessment", {})
        
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
            
        return state
    
    @traceable(name="process_message")
    def process_message(self, user_input: str) -> Dict[str, Any]:
        initial_state = {
            "user_input": user_input,
            "security_check": {},
            "response": "",
            "should_block": False
        }
        
        result = self.workflow.invoke(initial_state)
        
        return {
            "response": result["response"],
            "security_check": result["security_check"],
            "blocked": result["should_block"],
            "classification": {
                "question_type": result.get("question_type"),
                "confidence": result.get("classification_confidence"),
                "reasoning": result.get("classification_reasoning"),
                "original_classification": result.get("original_classification")
            },
            "safety_assessment": result.get("safety_assessment", {})
        }
    
    def clear_history(self):
        self.chatbot.clear_history()
    
    def get_conversation_history(self):
        return self.chatbot.get_conversation_history()