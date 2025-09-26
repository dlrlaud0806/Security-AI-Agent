import time
import re
import os
from typing import Dict, Any, Optional, Tuple, List
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

class SAPSecurityAssessment(BaseModel):
    is_safe: bool = Field(description="SAP 작업 안전 여부")
    risk_level: str = Field(description="위험 등급: low, medium, high, critical")
    detected_patterns: List[str] = Field(description="감지된 위험 패턴", default=[])
    reasoning: str = Field(description="보안 평가 근거")
    blocked_reason: Optional[str] = Field(description="차단 사유", default=None)

class SAPAutomationAgent:
    def __init__(self, debug_mode: bool = None):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.tracker = LangSmithTracker("sap_automation")
        self.sap_session = None
        self.security_parser = PydanticOutputParser(pydantic_object=SAPSecurityAssessment)
        
        # 디버그 모드 설정 (환경변수 또는 매개변수로 설정 가능)
        if debug_mode is not None:
            self.debug_mode = debug_mode
        else:
            self.debug_mode = os.getenv('SAP_DEBUG_MODE', 'false').lower() == 'true'
        
        # SAP 위험 패턴 정의
        self.dangerous_patterns = {
            'financial_operations': [
                r'전표.*금액.*0', r'amount.*0', r'금액.*변경', r'amount.*change',
                r'전표.*삭제', r'document.*delete', r'회계.*삭제', r'accounting.*delete',
                r'자산.*삭제', r'asset.*delete', r'매출.*변경', r'revenue.*change'
            ],
            'data_manipulation': [
                r'모든.*삭제', r'all.*delete', r'전체.*삭제', r'entire.*delete',
                r'테이블.*drop', r'table.*drop', r'데이터.*삭제', r'data.*delete',
                r'마스터.*변경', r'master.*change', r'기준.*변경', r'master.*modify'
            ],
            'system_operations': [
                r'시스템.*종료', r'system.*shutdown', r'서버.*종료', r'server.*shutdown',
                r'백업.*삭제', r'backup.*delete', r'로그.*삭제', r'log.*delete',
                r'사용자.*삭제', r'user.*delete', r'권한.*변경', r'authority.*change'
            ],
            'batch_operations': [
                r'배치.*실행', r'batch.*run', r'대량.*변경', r'mass.*change',
                r'일괄.*변경', r'bulk.*change', r'전체.*업데이트', r'全て.*更新'
            ],
            'critical_transactions': [
                r'se80', r'se16', r'sm30', r'se11', r'se38', r'se93',  # 개발/설정 트랜잭션
                r'su01d', r'pfcg', r'su53', r'sm19',  # 권한 관련 위험 트랜잭션
                r'sm50', r'sm66', r'sm21', r'st22',   # 시스템 관리 트랜잭션
                r'f-.*', r'fb.*', r'fs.*'             # 재무 트랜잭션 (일부)
            ]
        }
        
        self.security_prompt = """당신은 SAP 시스템 보안 전문가입니다. 
사용자의 SAP 자동화 요청을 분석하여 보안 위험을 평가해주세요.

**위험 등급 기준:**
- **low**: 일반적인 조회/확인 작업
- **medium**: 단일 레코드 수정, 일반 사용자 작업
- **high**: 중요 데이터 변경, 시스템 설정 변경
- **critical**: 재무 데이터 변경, 대량 삭제/수정, 시스템 관리 작업

**차단해야 할 위험한 요청들:**
1. **재무 데이터 조작**: 전표금액 변경, 회계 데이터 삭제
2. **대량 데이터 조작**: 모든/전체 데이터 삭제, 일괄 변경
3. **시스템 관리**: 사용자 삭제, 권한 변경, 시스템 종료
4. **위험한 트랜잭션**: 개발 트랜잭션(SE80, SE16 등), 시스템 관리 트랜잭션

**허용 가능한 안전한 작업들:**
- 사용자 락 해제 (SU01에서 unlock만)
- 단순 조회 작업
- 개별 레코드 확인

{format_instructions}"""
    
    def _check_dangerous_patterns(self, request: str) -> List[str]:
        """위험한 패턴 검사"""
        detected_patterns = []
        
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request, re.IGNORECASE):
                    detected_patterns.append(f"{category}: {pattern}")
        
        return detected_patterns
    
    @traceable(name="assess_sap_security")
    def assess_sap_security(self, request: str) -> SAPSecurityAssessment:
        """SAP 요청의 보안 위험을 평가"""
        try:
            # 먼저 패턴 기반 검사 수행
            pattern_issues = self._check_dangerous_patterns(request)
            
            messages = [
                SystemMessage(content=self.security_prompt.format(
                    format_instructions=self.security_parser.get_format_instructions()
                )),
                HumanMessage(content=f"다음 SAP 자동화 요청의 보안 위험을 평가해주세요:\n\n{request}")
            ]
            
            response = self.llm.invoke(messages)
            result = self.security_parser.parse(response.content)
            
            # 패턴 검사 결과를 LLM 결과와 결합
            if pattern_issues:
                result.detected_patterns.extend(pattern_issues)
                # 위험한 패턴이 감지되면 무조건 차단
                if result.risk_level in ["low", "medium"]:
                    result.risk_level = "critical"
                    result.is_safe = False
                    result.blocked_reason = f"위험한 패턴 감지: {', '.join(pattern_issues)}"
            
            # critical 등급은 무조건 차단
            if result.risk_level == "critical":
                result.is_safe = False
                if not result.blocked_reason:
                    result.blocked_reason = "고위험 SAP 작업으로 분류되어 차단"
            
            return result
            
        except Exception as e:
            return SAPSecurityAssessment(
                is_safe=False,
                risk_level="critical",
                detected_patterns=["system_error"],
                reasoning=f"보안 평가 중 오류 발생: {str(e)}",
                blocked_reason="시스템 오류로 인한 안전 차단"
            )
    
    def _handle_unlock_popup(self, session, username: str, dispatcher=None) -> SAPAutomationResult:
        """SAP 사용자 락 해제 팝업 처리"""
        try:
            # 팝업 창 정보 수집
            popup_info = self._extract_popup_info(session)
            
            if self.debug_mode:
                print(f"[DEBUG] 팝업 제목: '{popup_info['title']}'")
                print(f"[DEBUG] 팝업 텍스트: '{popup_info['text']}'")  
                print(f"[DEBUG] 버튼 개수: {popup_info['button_count']}")
                print(f"[DEBUG] 모든 텍스트: {popup_info['all_texts']}")
            
            # 락 상태 판별 로직 (우선순위 기반)
            is_locked = self._determine_lock_status(
                popup_info['title'], 
                popup_info['text'], 
                popup_info['button_count'],
                popup_info['all_texts']
            )
            
            if is_locked == "not_locked":
                # 잠겨있지 않음 - Cancel(F12) 실행
                session.findById("wnd[1]").sendVkey(12)  # F12 키 (Cancel)
                if dispatcher:
                    dispatcher.utter_message(text=f"ℹ️ {username} 계정은 잠겨 있지 않아요.")
                return SAPAutomationResult(
                    success=True,
                    message=f"사용자 '{username}'는 잠겨 있지 않습니다.",
                    details="사용자가 잠금 상태가 아닙니다."
                )
            elif is_locked == "locked":
                # 잠겨있음 - Yes(F7) 실행하여 잠금 해제
                session.findById("wnd[1]").sendVkey(7)  # F7 키 (Yes)
                if dispatcher:
                    dispatcher.utter_message(text=f"✅ {username} 계정의 잠금을 해제했어요!")
                return SAPAutomationResult(
                    success=True,
                    message=f"사용자 '{username}' 락 해제 완료",
                    details=f"사용자 {username}의 잠금을 성공적으로 해제했습니다."
                )
            else:
                # 알 수 없는 상태 - Cancel로 안전하게 종료
                session.findById("wnd[1]").sendVkey(12)  # F12 키 (Cancel)
                
                details = f"""
팝업 정보 분석:
• 제목: '{popup_info['title']}' 
• 텍스트: '{popup_info['text']}'
• 버튼 개수: {popup_info['button_count']}

수집된 모든 텍스트:
{chr(10).join(popup_info['all_texts']) if popup_info['all_texts'] else '텍스트를 찾을 수 없습니다'}

해결 방법:
1. SAP GUI에서 수동으로 확인
2. 팝업 창의 정확한 필드명 확인 후 코드 수정
3. SAP 시스템 언어 설정 확인"""

                return SAPAutomationResult(
                    success=False,
                    message="사용자 잠금 상태를 확인할 수 없습니다.",
                    details=details
                )
                
        except Exception as e:
            return SAPAutomationResult(
                success=False,
                message="팝업 처리 중 오류 발생",
                details=f"오류 내용: {str(e)}"
            )
    
    def _extract_popup_info(self, session) -> dict:
        """팝업창에서 모든 가능한 정보를 추출"""
        popup_info = {
            'title': '',
            'text': '',
            'button_count': 0,
            'all_texts': []
        }
        
        # 1. 팝업 제목 시도
        title_fields = [
            "wnd[1]/titl",
            "wnd[1]/sbar",
            "wnd[1]/usr/lblG_TITLE"
        ]
        
        for field in title_fields:
            try:
                title = session.findById(field).Text
                if title and title.strip():
                    popup_info['title'] = title.lower().strip()
                    popup_info['all_texts'].append(f"제목: {title}")
                    break
            except:
                continue
        
        # 2. 팝업 텍스트 수집 (모든 가능한 텍스트 필드)
        text_fields = [
            # 일반적인 텍스트 필드들
            "wnd[1]/usr/txtG_STATTEXT",
            "wnd[1]/usr/txtSPOP-TEXTLINE1", 
            "wnd[1]/usr/txtSPOP-TEXTLINE2",
            "wnd[1]/usr/txtSPOP-TEXTLINE3",
            
            # 라벨들
            "wnd[1]/usr/lbl[0,3]",
            "wnd[1]/usr/lbl[0,4]", 
            "wnd[1]/usr/lbl[0,5]",
            
            # 상태바
            "wnd[1]/sbar/pane[0]",
            
            # 기타 가능한 필드들
            "wnd[1]/usr/txt[0,3]",
            "wnd[1]/usr/txt[0,4]",
            "wnd[1]/usr/txt[0,5]"
        ]
        
        collected_texts = []
        for field in text_fields:
            try:
                text = session.findById(field).Text
                if text and text.strip():
                    collected_texts.append(text.strip())
                    popup_info['all_texts'].append(f"{field}: {text}")
            except:
                continue
        
        popup_info['text'] = ' '.join(collected_texts).lower()
        
        # 3. 버튼 개수
        try:
            popup_info['button_count'] = session.findById("wnd[1]/tbar[0]").Children.Count
        except:
            popup_info['button_count'] = 0
        
        # 4. 팝업의 모든 자식 요소 스캔 (추가 정보)
        try:
            self._scan_popup_children(session, popup_info)
        except:
            pass
            
        return popup_info
    
    def _scan_popup_children(self, session, popup_info):
        """팝업의 모든 자식 요소를 스캔하여 추가 텍스트 수집"""
        try:
            popup = session.findById("wnd[1]")
            
            # usr 컨테이너의 모든 자식 요소 확인
            usr = popup.findById("usr")
            if usr:
                children_count = usr.Children.Count if hasattr(usr, 'Children') else 0
                popup_info['all_texts'].append(f"usr 자식 요소 개수: {children_count}")
                
                # 처음 10개 자식 요소의 텍스트 수집
                for i in range(min(10, children_count)):
                    try:
                        child = usr.Children.Item(i)
                        if hasattr(child, 'Text') and child.Text:
                            popup_info['all_texts'].append(f"자식[{i}]: {child.Text}")
                    except:
                        continue
        except:
            pass
    
    def _determine_lock_status(self, popup_title: str, popup_text: str, button_count: int, all_texts: list = None) -> str:
        """다양한 조건을 종합하여 락 상태 판별"""
        
        # 0. 모든 텍스트를 하나로 결합 (all_texts 포함)
        all_combined_text = f"{popup_title} {popup_text}".lower()
        
        if all_texts:
            for text_item in all_texts:
                all_combined_text += f" {text_item}".lower()
        
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"[DEBUG] 결합된 전체 텍스트: '{all_combined_text}'")
        
        # 1. 텍스트 기반 우선 판별 (가장 정확)
        locked_keywords = [
            "locked", "잠김", "잠겨", "lock", "차단", "blocked",
            "unlock user", "사용자 잠금해제", "잠금을 해제",
            "locked by", "system manager", "관리자", "administrator"
        ]
        
        not_locked_keywords = [
            "not locked", "잠겨있지 않", "잠금되지 않", "unlocked",
            "활성", "active", "정상", "not locked", "no lock"
        ]
        
        # 명확한 "잠겨있지 않음" 키워드가 있는 경우
        for keyword in not_locked_keywords:
            if keyword in all_combined_text:
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"[DEBUG] 'not locked' 키워드 발견: '{keyword}'")
                return "not_locked"
        
        # 명확한 "잠김" 키워드가 있는 경우  
        for keyword in locked_keywords:
            if keyword in all_combined_text:
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"[DEBUG] 'locked' 키워드 발견: '{keyword}'")
                return "locked"
        
        # 2. 팝업 제목 기반 판별
        if "unlock" in popup_title or "잠금해제" in popup_title:
            return "locked"
        
        # 3. 버튼 개수 기반 보조 판별 (텍스트가 명확하지 않을 때만)
        if button_count == 2:
            # Yes/No 버튼 - 보통 잠금해제 확인
            return "locked" 
        elif button_count == 1:
            # OK 버튼만 - 보통 정보 표시
            return "not_locked"
        elif button_count == 3:
            # 3개 버튼 - 보통 잠겨있지 않은 상태에서 OK/Cancel 등
            # 하지만 텍스트가 없으면 확신할 수 없음
            return "unknown"
        
        # 4. 알 수 없음
        return "unknown"
        
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
                if "does not exist" in message or "존재하지 않습니다" in message:
                    if dispatcher:
                        dispatcher.utter_message(text=f"❗ 사용자 {username}는 존재하지 않아요.")
                    return SAPAutomationResult(
                        success=False,
                        message=f"사용자 {username}는 존재하지 않습니다.",
                        details="존재하지 않는 사용자입니다."
                    )
            else:
                return self._handle_unlock_popup(session, username, dispatcher)
                    
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
            # 1. 보안 검증 먼저 수행
            security_assessment = self.assess_sap_security(request)
            
            if not security_assessment.is_safe:
                return SAPAutomationResult(
                    success=False,
                    message="❌ 보안 위험으로 인한 작업 차단",
                    details=f"""
보안 평가 결과:
• 위험 등급: {security_assessment.risk_level}
• 차단 사유: {security_assessment.blocked_reason}
• 감지된 패턴: {', '.join(security_assessment.detected_patterns) if security_assessment.detected_patterns else '없음'}
• 평가 근거: {security_assessment.reasoning}

안전한 SAP 작업만 요청해주세요. (예: 사용자 락 해제)
"""
                )
            
            # 2. 보안 검증 통과 시에만 실제 작업 수행
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
            
            # 요청 유형 분석 (현재는 안전한 unlock 작업만 지원)
            if any(keyword in request.lower() for keyword in ['락해제', '잠금해제', 'unlock']):
                return self.unlock_user(username, dispatcher=None)
            else:
                return SAPAutomationResult(
                    success=False,
                    message="지원하지 않는 작업입니다.",
                    details=f"""
현재 지원되는 안전한 작업: 사용자 락 해제

보안 평가 결과:
• 위험 등급: {security_assessment.risk_level}
• 평가 근거: {security_assessment.reasoning}
"""
                )
                
        except Exception as e:
            return SAPAutomationResult(
                success=False,
                message="요청 처리 중 오류 발생",
                details=f"오류 내용: {str(e)}"
            )