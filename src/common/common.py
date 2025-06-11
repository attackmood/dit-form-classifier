# src/common/common.py

import logging
from datetime import datetime
from config import config


def get_timestamp() -> str:
    """
    현재 시간을 'YYYYMMDD_HHMMSS' 형식의 문자열로 반환합니다.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    애플리케이션 전반에 걸쳐 사용할 로거를 설정합니다.
    로그 파일은 config.LOG_DIR_FOR_APP에 저장됩니다.

    Args:
        name (str): 로거의 이름 (일반적으로 __name__을 사용).

    Returns:
        logging.Logger: 설정된 로거 인스턴스.
    """
    # config에서 로그 디렉토리와 레벨을 가져옵니다.
    log_dir = config.LOG_DIR_FOR_APP
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)  # config의 LOG_LEVEL 사용

    # 로그 디렉토리 생성
    # Path 객체에 직접 mkdir 메서드를 사용하고 parents=True로 상위 디렉토리까지 생성
    log_dir.mkdir(parents=True, exist_ok=True)

    # 로거 인스턴스 생성
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 중복 핸들러 방지: 로거에 핸들러가 이미 추가되었는지 확인
    if not logger.handlers:
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        # 콘솔에는 INFO 레벨 이상의 로그를 출력하도록 설정 (원한다면 변경 가능)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        # 파일 핸들러 설정
        timestamp = get_timestamp()
        # Path 객체 연산자를 사용하여 로그 파일 경로를 안전하게 구성
        log_file_path = log_dir / f"app_{timestamp}.log"  # 파일명을 좀 더 구체적으로 'app_YYYYMMDD_HHmmss.log'로 변경

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(log_level)  # 파일에는 config에 설정된 레벨로 기록
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger
