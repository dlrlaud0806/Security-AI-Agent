from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage
from ..agents.security_agent import PromptInjectionDetector
from .chatbot import Chatbot

class ChatbotState:
    def __init__(self):
        self.user_input: str = ""
        self.security_check: Dict[str, Any] = {}
        self.response: str = ""
        self.should_block: bool = False

class SecureChatbotWorkflow:
    def __init__(self, system_prompt: str = "You are a helpful AI assistant."):
        self.security_agent = PromptInjectionDetector()
        self.chatbot = Chatbot(system_prompt)
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(dict)
        
        workflow.add_node("security_check", self._security_check_node)
        workflow.add_node("process_message", self._process_message_node)
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
        
        workflow.add_edge("process_message", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
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
    
    def _process_message_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        sanitized_input = self.security_agent.sanitize_input(user_input)
        state["sanitized_input"] = sanitized_input
        return state
    
    def _generate_response_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_input = state.get("sanitized_input", "")
        response = self.chatbot.chat(sanitized_input)
        state["response"] = response
        return state
    
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
            "blocked": result["should_block"]
        }
    
    def clear_history(self):
        self.chatbot.clear_history()
    
    def get_conversation_history(self):
        return self.chatbot.get_conversation_history()