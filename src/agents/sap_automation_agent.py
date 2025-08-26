import time
import re
from typing import Dict, Any, Optional, Tuple
import win32com.client
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langsmith import traceable
from ..config.settings import settings
from ..utils.langsmith_config import LangSmithTracker


class SAPAutomationResult(BaseModel):
    success: bool = Field(description="작업 성공 여부")
    message: str = Field(description="작업 결과 메시지")
    details: str = Field(description="작업 상세 내용")

class SAPAutomationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.tracker = LangSmithTracker("sap_automation")
        self.sap_session = None
        
    def _connect_to_sap(self) -> bool:
        """SAP GUI Script 엔진에 연결합니다."""
        try:
            SapGuiAuto = win32com.client.GetObject("SAPGUI")
            application = SapGuiAuto.GetScriptingEngine
            connection = application.Children(0)
            session = connection.Children(0)
            self.sap_session = session
            return True
        except Exception as e:
            print(f"SAP GUI Script 연결 실패: {str(e)}")
            return False
    
    
    @traceable(name="unlock_user")
    def unlock_user(self, username: str, dispatcher=None) -> SAPAutomationResult:
        """SAP에서 사용자 락을 해제합니다."""
        try:
            # 1. SAP GUI Script 엔진에 연결
            if not self._connect_to_sap():
                return SAPAutomationResult(
                    success=False,
                    message="SAP GUI Script 연결 실패",
                    details="SAP GUI가 실행되어 있는지 확인하고 로그인된 상태인지 확인해주세요."
                )
            
            session = self.sap_session
            
            # 2. SU01 트랜잭션 실행
            session.StartTransaction(Transaction="SU01")
            time.sleep(1)

            # 3. 사용자명 입력
            session.findById("wnd[0]/usr/ctxtSUID_ST_BNAME-BNAME").text = username
            time.sleep(0.5)

            # 4. 잠금 해제 버튼 클릭
            session.findById("wnd[0]/tbar[1]/btn[29]").press()
            time.sleep(1)

            # 5. 팝업 창 처리
            if session.Children.Count == 1:
                message = session.findById("wnd[0]/sbar/pane[0]").Text
                if "does not exist" in message:
                    if dispatcher:
                        dispatcher.utter_message(text=f"❗ 사용자 {username}는 존재하지 않아요.")
                    return SAPAutomationResult(
                        success=False,
                        message=f"사용자 {username}는 존재하지 않습니다.",
                        details="존재하지 않는 사용자입니다."
                    )
            else:
                # 잠금 상태에 따라 메시지 처리
                button_count = session.findById("wnd[1]/tbar[0]").Children.Count
                
                # 상태 텍스트 확인
                status_text = ""
                try:
                    status_text = session.findById("wnd[1]/usr/txtG_STATTEXT").Text
                except:
                    pass
                
                # Not locked 조건: 버튼이 3개이거나 상태 텍스트에 "Not locked" 포함
                if button_count == 3 or "Not locked" in status_text:
                    # Not locked 상태 - Cancel(F12) 실행
                    session.findById("wnd[1]").sendVkey(12)  # F12 키 (Cancel)
                    if dispatcher:
                        dispatcher.utter_message(text=f"ℹ️ {username} 계정은 잠겨 있지 않아요.")
                    return SAPAutomationResult(
                        success=True,
                        message=f"사용자 '{username}'는 잠겨 있지 않습니다.",
                        details="사용자가 잠금 상태가 아닙니다."
                    )
                elif button_count == 2:
                    # Locked 상태 - Yes(F7) 실행하여 잠금 해제
                    session.findById("wnd[1]").sendVkey(7)  # F7 키 (Yes)
                    if dispatcher:
                        dispatcher.utter_message(text=f"✅ {username} 계정의 잠금을 해제했어요!")
                    return SAPAutomationResult(
                        success=True,
                        message=f"사용자 '{username}' 락 해제 완료",
                        details=f"사용자 {username}의 잠금을 성공적으로 해제했습니다."
                    )
                else:
                    message = session.findById("wnd[0]/sbar/pane[0]").Text if session.Children.Count == 1 else "알 수 없는 상태"
                    if dispatcher:
                        dispatcher.utter_message(text=f"🛈 SAP 메시지: {message}")
                    return SAPAutomationResult(
                        success=False,
                        message="예상치 못한 SAP 응답",
                        details=f"SAP 메시지: {message}"
                    )
                    
        except Exception as e:
            return SAPAutomationResult(
                success=False,
                message="사용자 락 해제 중 오류 발생",
                details=f"오류 내용: {str(e)}"
            )
    
    
    @traceable(name="process_user_request")
    def process_user_request(self, request: str) -> SAPAutomationResult:
        """사용자 요청을 분석하고 적절한 SAP 작업을 수행합니다."""
        try:
            # 사용자 이름 추출
            user_pattern = r'사용자\s+([A-Za-z0-9_]+)|user\s+([A-Za-z0-9_]+)|([A-Za-z0-9_]+)\s+사용자'
            user_match = re.search(user_pattern, request, re.IGNORECASE)
            
            if not user_match:
                return SAPAutomationResult(
                    success=False,
                    message="사용자 이름을 찾을 수 없습니다.",
                    details="요청에서 사용자 이름을 명시해주세요. 예: '사용자 TESTUSER 락해제 해주세요'"
                )
            
            username = user_match.group(1) or user_match.group(2) or user_match.group(3)
            username = username.upper()  # SAP 사용자명은 대문자로 처리
            
            # 요청 유형 분석
            if any(keyword in request.lower() for keyword in ['락해제', '잠금해제', 'unlock']):
                return self.unlock_user(username, dispatcher=None)
            else:
                return SAPAutomationResult(
                    success=False,
                    message="지원하지 않는 작업입니다.",
                    details="현재 지원되는 작업: 사용자 락 해제"
                )
                
        except Exception as e:
            return SAPAutomationResult(
                success=False,
                message="요청 처리 중 오류 발생",
                details=f"오류 내용: {str(e)}"
            )