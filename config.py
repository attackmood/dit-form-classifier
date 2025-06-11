# config.py

from pathlib import Path  # Pathlib 모듈 임포트


class DiTConfig:
    """DiT Fine-tuning 프로젝트의 모든 설정값을 정의합니다."""

    # 프로젝트 기본 경로 설정 ---
    PROJECT_ROOT = Path(__file__).resolve().parents[0]  # 현재 config.py가 프로젝트 루트에 있다면 parents[0]

    # --- 1. 데이터셋 설정 ---
    DATASET_BASE_DIR = "F:\\Datasets\\request_form\\model_train_data"
    CLASS_NAMES = ['g-scanning', 'the-mom-scanning', 'others']
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # --- 2. 모델 설정 ---
    MODEL_NAME = "microsoft/dit-base"
    NUM_LABELS = len(CLASS_NAMES)  # CLASS_NAMES 길이에 따라 자동 설정

    # --- 3. 학습 설정 ---
    NUM_TRAIN_EPOCHS = 5
    PER_DEVICE_TRAIN_BATCH_SIZE = 8
    PER_DEVICE_EVAL_BATCH_SIZE = 8
    LOGGING_STEPS = 10
    SAVE_STRATEGY = "epoch"
    EVALUATION_STRATEGY = "epoch"
    LOAD_BEST_MODEL_AT_END = True
    METRIC_FOR_BEST_MODEL = "accuracy"
    NO_CUDA = True  # CPU 환경이므로 True
    REPORT_TO = "tensorboard"  # TensorBoard를 사용하지 않으려면 "" 또는 None

    # --- 4. 프로젝트 내부 저장 경로 ---
    # 훈련 관련 아웃풋 디렉토리
    TRAINING_OUTPUT_BASE_DIR = PROJECT_ROOT / "runs"

    # 모델 체크포인트, 로그, 최종 모델 저장 경로
    LOGGING_DIR = TRAINING_OUTPUT_BASE_DIR / "dit_logs"
    OUTPUT_DIR = TRAINING_OUTPUT_BASE_DIR / "dit_results"
    SAVE_MODEL_PATH = TRAINING_OUTPUT_BASE_DIR / "fine_tuned_dit_classifier"

    # 훈련 진행 상황 이미지 저장 경로
    SAVE_PLOT_DIR = PROJECT_ROOT / "training_progress_images"

    # 예측 결과 저장 경로
    PRED_SAVE_PATH = PROJECT_ROOT / "predictions"

    # 로깅 파일 저장 경로 (common.py의 setup_logger에서 사용)
    LOG_DIR_FOR_APP = PROJECT_ROOT / "logs"  # 예: dit-form-classifier/logs
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- 5. 임계점 설정 ---
    THRESHOLD = 0.82

    # --- 6. 프로세싱 설정 (datasets.map() 에 사용) ---
    NUM_PROC_DATASET_MAP = 1  # 멀티프로세싱을 사용하려면 CPU 코어 수로 설정 (예: os.cpu_count())

    # --- 7. 실제 예측 시 필요한 모델 경로 ---
    # 훈련 관련 아웃풋 디렉토리
    FINE_TUNED_MODEL_PATH = PROJECT_ROOT / "saved_models" / "fine_tuned_dit_classifier"

    def __init__(self):
        # 클래스 인스턴스 생성 시 필요한 디렉토리들을 생성합니다.
        # .mkdir(parents=True, exist_ok=True)를 사용하여 안전하게 생성

        self.TRAINING_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGGING_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 모델 체크포인트
        self.SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)  # 최종 저장 모델
        self.SAVE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
        self.PRED_SAVE_PATH.mkdir(parents=True, exist_ok=True)  # 예측 결과 저장 폴더
        self.LOG_DIR_FOR_APP.mkdir(parents=True, exist_ok=True)  # 애플리케이션 로그 폴더


config = DiTConfig()
