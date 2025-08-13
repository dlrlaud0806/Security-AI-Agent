import httpx
from typing import Optional
from pydantic_settings import BaseSettings
import ssl
import httpx
# # SSL 검증 비활성화
# ssl_context = ssl.create_default_context()
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_NONE

# # httpx 클라이언트 설정
# skipsslclient = httpx.Client(verify=False)
# openai_api_key: Optional[str] = None
# mistral_api_key: Optional[str] = None
class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # LangSmith 설정
    langsmith_tracing: Optional[str] = None
    langsmith_endpoint: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
    
    # client: httpx.Client = skipsslclient
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()