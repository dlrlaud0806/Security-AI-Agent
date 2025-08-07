from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
# from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from ..config.settings import settings

# #vdi
# import ssl
# import httpx
# # SSL 검증 비활성화
# ssl_context = ssl.create_default_context()
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_NONE

# httpx 클라이언트 설정
# skipsslclient = httpx.Client(verify=False)

class Chatbot:
    def __init__(self, system_prompt: str = "You are a helpful AI assistant."):
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        # self.llm = ChatMistralAI(
        #     model="mistral-large-latest",
        #     temperature=0,
        #     max_retries=2,
        #     client=skipsslclient
        #     # other params...
        # )
        self.memory = ChatMessageHistory()
        self.system_prompt = system_prompt
    
    def chat(self, message: str) -> str:
        messages = [SystemMessage(content=self.system_prompt)]
        
        chat_history = self.memory.messages
        messages.extend(chat_history)
        
        messages.append(HumanMessage(content=message))
        
        response = self.llm(messages)
        
        self.memory.add_user_message(message)
        self.memory.add_ai_message(response.content)
        
        return response.content
    
    def clear_history(self):
        self.memory.clear()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        history = []
        for message in self.memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history