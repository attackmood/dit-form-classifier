# train.py

from src.common.common import setup_logger
from src.data.dataset import create_dit_dataset
from src.models.trainer import DiTModelTrainer

# 로거 설정
logger = setup_logger(__name__)

def run_fine_tuning_workflow():
    """전체 Fine-tuning 워크플로우를 실행합니다 (학습 및 저장)."""
    logger.info("--- DiT Fine-tuning 워크플로우 시작 ---")

    # 1. 데이터셋 준비 및 전처리
    train_dataset, test_dataset, label_to_id, id_to_label, processor = create_dit_dataset()

    if train_dataset is None or test_dataset is None:
        logger.error("데이터셋 준비 중 오류가 발생하여 학습을 중단합니다.")
        return

    # 2. 모델 학습 및 평가
    trainer_instance = DiTModelTrainer(train_dataset, test_dataset, label_to_id, id_to_label)
    trainer_instance.load_model()           # DiT 모델 로드 및 Fine-tuning 준비
    trainer_instance.configure_trainer()    # TrainingArguments 및 Trainer 설정
    trainer_instance.train_model()          # 모델 학습 (Fine-tuning)
    trainer_instance.evaluate_model()       # 모델 평가 (Fine-tuning)
    trainer_instance.save_model(processor)  # 학습된 모델 저장 : Fine-tuning된 모델과 프로세서를 저장
    trainer_instance.plot_training_progress() # 학습 진행 상황 그래프 저장

    logger.info("--- DiT Fine-tuning 워크플로우 완료 ---")

if __name__ == '__main__':
    run_fine_tuning_workflow()