"""
Report Tool - ESG 보고서 데이터 통합 및 생성 인터페이스
------------------------------------------------------

정책·리스크 데이터 저장 및 HTML/PDF 출력

주요 기능:
- 데이터 자동 로드: 상위 폴더에서 JSON 파일 자동 검색
- 대화형 입력 모드: 터미널을 통한 누락 데이터 입력
- 데이터 검증: 필수 필드 및 유효성 검사
- 보고서 생성: HTML/PDF 형식 보고서 출력

버전: 1.0.0
작성일: 2025-12-09
"""

import shutil
import platform
import subprocess
import os
import json
from typing import Any, Dict, List, Optional
from .esg_report_generator import generate_esg_report

class DataLoader:
    """데이터 자동 로드 및 검색
    
    상위 폴더 계층에서 데이터 파일을 자동으로 검색하고 로드합니다.
    """
    
    @staticmethod
    def find_and_load(filename: str = "esg_data.json") -> Dict[str, Any]:
        """상위 폴더에서 데이터 파일 검색 및 로드
        
        Parameters
        ----------
        filename : str, optional
            검색할 파일명 (기본값: "esg_data.json")
        
        Returns
        -------
        Dict[str, Any]
            로드된 JSON 데이터 (파일이 없으면 빈 딕셔너리)
        
        Notes
        -----
        검색 순서: 현재 폴더 -> 상위 폴더 -> 상위의 상위 폴더
        """
        # 검색 경로: 현재 폴더 -> 상위 -> 상위의 상위
        search_paths = [
            os.path.join(".", filename),
            os.path.join("..", filename),
            os.path.join("..", "..", filename)
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        print(f"[Info] 데이터 파일 발견: {os.path.abspath(path)}")
                        return data
                except json.JSONDecodeError as e:
                    print(f"[Warning] JSON 파싱 실패 ({path}): {e}")
                except Exception as e:
                    print(f"[Warning] 파일 로드 실패 ({path}): {e}")
        
        print("[Info] 자동 로드할 데이터 파일이 없습니다.")
        return {}

class ReportTool:
    """ESG 보고서 생성 도구
    
    데이터 로드, 검증, 저장 및 보고서 생성의 전 과정을 관리합니다.
    
    Attributes
    ----------
    _data : Dict[str, Any]
        내부 데이터 저장소 (private)
    
    Examples
    --------
    >>> tool = ReportTool()
    >>> tool.load_from_file("esg_data.json")
    >>> tool.create_report(report_path="esg_report.html")
    """
    
    def __init__(self):
        """ReportTool 초기화
        
        빈 딕셔너리로 내부 데이터 저장소를 초기화합니다.
        """
        self._data: Dict[str, Any] = {}
        
    def store_data(self, data: Dict[str, Any]) -> None:
        """데이터 저장 (기존 데이터와 병합)
        
        Parameters
        ----------
        data : Dict[str, Any]
            저장할 데이터 (기존 키가 있으면 덮어씀)
        
        Notes
        -----
        - 기존 키가 있으면 값을 덮어씀
        - 새로운 키는 추가됨
        - 중첩된 딕셔너리는 얕은 병합(shallow merge)
        """
        self._data.update(data)

    def load_from_file(self, filename: str = "esg_data.json") -> None:
        """파일에서 데이터 자동 로드
        
        Parameters
        ----------
        filename : str, optional
            로드할 파일명 (기본값: "esg_data.json")
        
        Notes
        -----
        DataLoader.find_and_load()를 사용하여 상위 폴더에서 파일을 검색합니다.
        로드된 데이터는 기존 데이터와 병합됩니다.
        """
        loaded = DataLoader.find_and_load(filename)
        if loaded:
            self.store_data(loaded)
            print(f" -> {len(loaded)}개 항목 로드됨")

    def gather_data_interactive(self) -> None:
        """대화형 데이터 입력 모드
        
        터미널을 통해 부족한 데이터를 사용자로부터 입력받습니다.
        
        Process
        -------
        1. load_from_file()을 호출하여 자동 로드 시도
        2. 필수 필드 확인 및 누락된 항목 입력 요청
        3. material_issues 누락 시 안내 메시지 출력
        
        Notes
        -----
        material_issues는 구조가 복잡하므로 JSON 파일이나 코드로 입력을 권장합니다.
        """
        print("\n[Interactive Mode] 부족한 데이터를 입력받습니다.")
        
        # 1. 자동 로드 시도
        self.load_from_file()
        
        # 2. 필수 필드 확인 및 요청
        required_fields = {
            "company_name": "회사명",
            "report_year": "보고 연도",
            "ceo_message": "CEO 메시지",
            "esg_strategy": "ESG 전략",
            "env_policy": "환경 정책",
            "social_policy": "사회 정책",
            "gov_structure": "지배구조"
        }
        
        for key, label in required_fields.items():
            if key not in self._data:
                val = input(f" > {label} 입력: ").strip()
                if val:
                    self._data[key] = val
                    
        # 3. 중대 이슈 (복잡한 데이터는 안내만)
        if "material_issues" not in self._data:
            print("\n[!] 'material_issues'(중대 이슈) 데이터가 없습니다.")
            print("    이 데이터는 구조가 복잡하므로 코드나 JSON 파일로 입력을 권장합니다.")
            print("    예시: [{'name': '기후변화', 'impact': 80, 'isMaterial': True}]")
    
    def get_data(self) -> Dict[str, Any]:
        """저장된 데이터 반환
        
        Returns
        -------
        Dict[str, Any]
            저장된 데이터의 복사본 (얕은 복사)
        
        Notes
        -----
        dict() 생성자로 새로운 딕셔너리를 생성하여 원본 데이터를 보호합니다.
        외부에서 반환된 딕셔너리를 수정해도 내부 데이터에 영향을 주지 않습니다.
        """
        return dict(self._data)
    
    def missing_fields(self) -> List[str]:
        """필수 필드 및 데이터 유효성 확인
        
        Returns
        -------
        List[str]
            검증 오류 목록 (오류가 없으면 빈 리스트)
        
        Validation Rules
        ----------------
        1. 필수 필드 검사:
           - company_name, report_year, ceo_message, esg_strategy
           - env_policy, social_policy, gov_structure
        
        2. material_issues 유효성 검사:
           - impact: 0-100 범위의 숫자
           - financial: 0-100 범위의 숫자
        
        Examples
        --------
        >>> tool = ReportTool()
        >>> tool.store_data({"company_name": "Test"})
        >>> errors = tool.missing_fields()
        >>> print(errors)
        ['누락된 필드: report_year', '누락된 필드: ceo_message', ...]
        """
        required = ["company_name", "report_year", "ceo_message", "esg_strategy", 
                   "env_policy", "social_policy", "gov_structure"]
        errors = [f"누락된 필드: {f}" for f in required if f not in self._data]
        
        # 데이터 유효성 검사
        if "material_issues" in self._data:
            issues = self._data["material_issues"]
            
            # material_issues가 리스트인지 확인
            if not isinstance(issues, list):
                errors.append("material_issues는 리스트 형식이어야 합니다.")
            else:
                for idx, issue in enumerate(issues):
                    if not isinstance(issue, dict):
                        errors.append(f"이슈 항목 {idx}는 딕셔너리 형식이어야 합니다.")
                        continue
                    
                    name = issue.get("name", f"Item {idx}")
                    
                    # Impact/Financial 점수 검사 (0-100)
                    for field in ["impact", "financial"]:
                        val = issue.get(field)
                        if val is not None:
                            if not isinstance(val, (int, float)):
                                errors.append(f"이슈 '{name}': {field}는 숫자여야 합니다.")
                            elif not (0 <= val <= 100):
                                errors.append(f"이슈 '{name}': {field}는 0-100 사이여야 합니다 ({val}).")

        return errors
    
    def _get_pdf_tools(self) -> Dict[str, str]:
        """PDF 변환에 필요한 도구 경로 확인
        
        Returns
        -------
        Dict[str, str]
            도구 이름과 경로 매핑 {"pandoc": "/path/to/pandoc", "libreoffice": "/path/to/soffice"}
        
        Raises
        ------
        RuntimeError
            필수 도구(Pandoc 또는 LibreOffice)를 찾을 수 없는 경우
        
        Notes
        -----
        검색 순서:
        1. Pandoc: shutil.which("pandoc")
        2. LibreOffice:
           - Linux: shutil.which("libreoffice") 또는 shutil.which("soffice")
           - Windows: PATH 검색 후 기본 설치 경로 확인
             * C:\\Program Files\\LibreOffice\\program\\soffice.exe
             * C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe
        """
        tools = {}
        
        # 1. Pandoc 확인
        pandoc = shutil.which("pandoc")
        if not pandoc:
            raise RuntimeError(
                "Pandoc을 찾을 수 없습니다. 설치 후 PATH에 추가해주세요.\n"
                "Windows: winget install JohnMacFarlane.Pandoc\n"
                "Linux: sudo apt-get install pandoc"
            )
        tools["pandoc"] = pandoc
        
        # 2. LibreOffice 확인
        # Windows: soffice, Linux: libreoffice/soffice
        libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
        
        # Windows의 일반적인 설치 경로 확인 (PATH에 없을 경우)
        if not libreoffice and platform.system() == "Windows":
             common_paths = [
                 r"C:\Program Files\LibreOffice\program\soffice.exe",
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
             ]
             for path in common_paths:
                 if os.path.exists(path):
                     libreoffice = path
                     break
        
        if not libreoffice:
            raise RuntimeError(
                "LibreOffice를 찾을 수 없습니다.\n"
                "Windows: https://www.libreoffice.org/download/download-libreoffice/ 설치 필요\n"
                "Linux: sudo apt-get install libreoffice"
            )
        tools["libreoffice"] = libreoffice
        
        return tools

    def create_report(self, user_inputs: Dict[str, Any] = None, 
                     report_path: Optional[str] = None) -> str:
        """ESG 보고서 생성 (HTML)
        
        Parameters
        ----------
        user_inputs : Dict[str, Any], optional
            추가 데이터 (기존 데이터와 병합됨)
        report_path : str, optional
            저장 경로 (.html 추천, .pdf 확장자도 지원하지만 HTML로 저장됨)
        
        Returns
        -------
        str
            생성된 HTML 보고서 전체 내용
        
        Process
        -------
        1. 데이터 준비: 내부 저장소 데이터 복사 및 user_inputs 병합
        2. 유효성 검증: missing_fields() 호출 및 경고 출력
        3. 보고서 생성: generate_esg_report(data) 호출
        4. 파일 저장: report_path가 지정된 경우 파일로 저장
        5. HTML 문자열 반환
        
        Notes
        -----
        - PDF 직접 변환은 지원하지 않으며, HTML 파일 생성만 수행합니다.
        - 향후 PDF 변환 방법:
          * WeasyPrint: HTML(string=html).write_pdf(output_path)
          * Pandoc: pandoc input.html -o output.pdf
          * Chromium: chromium --headless --print-to-pdf=output.pdf input.html
        
        Examples
        --------
        >>> tool = ReportTool()
        >>> tool.load_from_file()
        >>> html = tool.create_report(report_path="esg_report.html")
        >>> print(f"보고서 길이: {len(html)} bytes")
        """
        # 데이터 병합
        data = self.get_data()
        if user_inputs:
            data.update(user_inputs)
            
        # 유효성 검사 경고
        validation_errors = self.missing_fields()
        if validation_errors:
            print("[Warning] 데이터 유효성 문제:")
            for err in validation_errors:
                print(f" - {err}")
        
        # HTML 보고서 생성
        try:
            report_html = generate_esg_report(data)
        except Exception as e:
            print(f"[Error] 보고서 생성 중 오류 발생: {e}")
            raise
        
        # 파일 저장
        if report_path:
            ext = os.path.splitext(report_path)[1].lower()
            
            if ext == ".pdf":
                # PDF 변환 안내
                print("[Notice] PDF 변환 기능은 HTML -> PDF 엔진이 필요합니다.")
                print("         현재 버전에서는 HTML 파일 저장을 권장합니다.")
                print("         브라우저에서 HTML 파일을 열어 '인쇄 -> PDF로 저장'을 사용하세요.")
                
                # HTML 파일도 같이 저장
                html_path = report_path.replace(".pdf", ".html")
                try:
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(report_html)
                    print(f"[Success] HTML 원본 저장: {html_path}")
                except Exception as e:
                    print(f"[Error] 파일 저장 실패: {e}")
                    raise
                
            else:
                # HTML 저장
                if not report_path.endswith(".html"):
                    report_path += ".html"
                
                try:
                    # 디렉토리가 없으면 생성
                    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
                    
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(report_html)
                    print(f"[Success] HTML 리포트 생성 완료: {report_path}")
                    print(f"          파일 크기: {len(report_html):,} bytes")
                except Exception as e:
                    print(f"[Error] 파일 저장 실패: {e}")
                    raise
        
        return report_html
