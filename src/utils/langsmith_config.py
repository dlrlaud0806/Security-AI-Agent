import os
import time
import uuid
from typing import Dict, Any, Optional
from langsmith import traceable
from ..config.settings import settings

def setup_langsmith():
    """LangSmith 환경변수 설정"""
    # 기존 환경변수 우선, 없으면 settings에서 가져오기
    if settings.langsmith_tracing and not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = settings.langsmith_tracing
    if settings.langsmith_endpoint and not os.getenv("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_api_key and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    
    # 기존 환경변수들을 다시 읽어서 출력 (디버깅용)
    print(f"LangSmith 설정 완료:")
    print(f"  TRACING: {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
    print(f"  PROJECT: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print(f"  ENDPOINT: {os.getenv('LANGCHAIN_ENDPOINT', 'default')}")
    print(f"  API_KEY: {'SET' if os.getenv('LANGCHAIN_API_KEY') else 'NOT_SET'}")
    
    return os.getenv("LANGCHAIN_TRACING_V2") == "true"

def get_langsmith_metadata(
    component_type: str,
    component_name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """LangSmith 추적용 메타데이터 생성"""
    metadata = {
        "component_type": component_type,
        "component_name": component_name,
        "project": "chatbot-security-system",
        "timestamp": time.time(),
        "run_id": str(uuid.uuid4()),
        "version": "1.0.0",
        "environment": "development"
    }
    
    if user_id:
        metadata["user_id"] = user_id
    if session_id:
        metadata["session_id"] = session_id
    
    metadata.update(kwargs)
    return metadata

def create_run_tags(component: str, operation: str, **extras) -> list:
    """실행 태그 생성"""
    tags = [
        f"component:{component}",
        f"operation:{operation}",
        "version:1.0",
        "system:chatbot-security",
        "environment:development"
    ]
    
    for key, value in extras.items():
        tags.append(f"{key}:{value}")
    
    return tags

def create_security_metadata(
    input_text: str,
    threat_level: Optional[str] = None,
    detected_patterns: Optional[list] = None
) -> Dict[str, Any]:
    """보안 관련 메타데이터 생성"""
    metadata = {
        "input_length": len(input_text),
        "input_hash": hash(input_text) % 1000000,  # 간단한 해시
        "security_scan": True
    }
    
    if threat_level:
        metadata["threat_level"] = threat_level
    if detected_patterns:
        metadata["detected_patterns"] = detected_patterns
    
    return metadata

def create_classification_metadata(
    question_type: str,
    confidence: float,
    fallback_used: bool = False
) -> Dict[str, Any]:
    """분류 관련 메타데이터 생성"""
    return {
        "classification_type": question_type,
        "confidence_score": confidence,
        "fallback_used": fallback_used,
        "classification_system": "llm_agent"
    }

class LangSmithTracker:
    """LangSmith 추적을 위한 헬퍼 클래스"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        setup_langsmith()
    
    @traceable
    def track_operation(
        self,
        operation_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None
    ):
        """작업 추적"""
        if metadata is None:
            metadata = {}
        
        if tags is None:
            tags = []
        
        # 기본 메타데이터 추가
        full_metadata = get_langsmith_metadata(
            component_type="agent",
            component_name=self.component_name,
            operation=operation_name,
            **metadata
        )
        
        # 기본 태그 추가
        full_tags = create_run_tags(
            component=self.component_name,
            operation=operation_name
        ) + tags
        
        return {
            "inputs": inputs,
            "outputs": outputs,
            "metadata": full_metadata,
            "tags": full_tags
        }