# trainer.py
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, EvalPrediction

from config import config
from src.common.common import get_timestamp, setup_logger

# 로거 설정
logger = setup_logger(__name__)

class DiTModelTrainer:
    """
    DiT 모델 학습 및 평가를 담당하는 클래스.
    """
    def __init__(self, train_dataset, test_dataset, label_to_id, id_to_label):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        self.model = None
        self.trainer = None
        self.progress_images_dir = config.SAVE_PLOT_DIR


    # --- 3단계: DiT 모델 로드 및 Fine-tuning 준비 ---
    def load_model(self):
        logger.info("\n--- 3단계: DiT 모델 로드 및 Fine-tuning 준비 시작 ---")
        self.model = AutoModelForImageClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS,
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        logger.info("DiT 모델 로드 완료.")

    # --- 4단계: TrainingArguments 및 Trainer 설정 ---
    def configure_trainer(self):
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=config.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
            logging_dir=config.LOGGING_DIR,
            logging_steps=config.LOGGING_STEPS,
            save_strategy=config.SAVE_STRATEGY,
            evaluation_strategy=config.EVALUATION_STRATEGY,
            load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
            no_cuda=config.NO_CUDA,
            seed=config.RANDOM_STATE,
            report_to=config.REPORT_TO
        )
        logger.info("\n--- 4단계: TrainingArguments 설정 완료 ---")

        # evaluation 시점에서 모델의 성능을 나타내는 지표를 계산하기 위한 함수
        # input : EvalPrediction / outputs: 정확도, 정밀도, 재현율, F1-Score
        def compute_metrics(p: EvalPrediction):
            predictions = np.argmax(p.predictions, axis=1) # 모델이 각 샘플에 대해 가장 높은 확률(또는 로짓)을 부여한 클래스의 인덱스를 반환
            # 분류 모델의 핵심 성능 지표를 계산하는 함수 : 다중 클래스일 경우 average='weighted' 추천
            precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, predictions, average='weighted', zero_division=0)
            acc = accuracy_score(p.label_ids, predictions)
            return {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        # Trainer 인스턴스 생성
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=compute_metrics,
        )
        logger.info("\n--- 4단계: Trainer 인스턴스 생성 완료 ---")

    # --- 5단계: 모델 학습 (Fine-tuning) ---
    def train_model(self):

        if self.trainer is None:
            raise ValueError("Trainer가 설정되지 않았습니다. configure_trainer()를 먼저 호출하세요.")

        logger.info("\n--- 5단계: 모델 학습 시작 (CPU에서 진행되므로 시간이 오래 걸릴 수 있습니다) ---")
        train_results = self.trainer.train()
        logger.info("\n--- 5단계: 모델 학습 완료 ---")
        logger.info(f"\n훈련 결과: {train_results.metrics}")
        return train_results

    # --- 5단계: 모델 평가 (Fine-tuning) ---
    def evaluate_model(self):
        if self.trainer is None:
            raise ValueError("Trainer가 설정되지 않았습니다. configure_trainer()를 먼저 호출하세요.")

        eval_results = self.trainer.evaluate(self.test_dataset)
        logger.info(f"\n테스트 세트 최종 평가: {eval_results}")
        return eval_results

    # --- 6단계: 학습된 모델 저장 : Fine-tuning된 모델과 프로세서를 저장---
    def save_model(self, processor):
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        logger.info("\n--- 6단계: 학습된 모델 저장 ---")
        self.model.config.id2label = self.id_to_label
        self.model.config.label2id = self.label_to_id
        self.model.save_pretrained(config.SAVE_MODEL_PATH)
        processor.save_pretrained(config.SAVE_MODEL_PATH)

        with open(os.path.join(config.SAVE_MODEL_PATH, "label_mapping.json"), 'w') as f:
            json.dump({'id_to_label': self.id_to_label, 'label_to_id': self.label_to_id}, f)

        logger.info(f"\nFine-tuning된 모델이 '{config.SAVE_MODEL_PATH}' 폴더에 저장되었습니다.")


    def _plot_loss_progress(self, training_loss_logs, eval_metrics_logs, current_timestamp):
        """
        훈련 및 검증 손실 추이를 그래프로 그리고 저장합니다.
        """
        if not training_loss_logs and not eval_metrics_logs:
            logger.warning("손실 데이터를 찾을 수 없어 Loss 그래프를 그릴 수 없습니다.")
            return

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        if training_loss_logs:
            epochs_loss = [x['epoch'] for x in training_loss_logs]
            train_losses = [x['loss'] for x in training_loss_logs]
            ax.plot(epochs_loss, train_losses, label='Intermediate Training Loss (loss)', color='blue', marker='x', linestyle='--')

        if eval_metrics_logs:
            epochs_eval_loss = [x['epoch'] for x in eval_metrics_logs]
            eval_losses = [x['eval_loss'] for x in eval_metrics_logs]
            ax.plot(epochs_eval_loss, eval_losses, label='Validation Loss (eval_loss)', color='red', marker='o', linestyle='-')

        # 최종 훈련 손실 추가
        final_train_summary = [x for x in self.trainer.state.log_history if 'train_loss' in x and 'epoch' in x and x['epoch'] == self.trainer.args.num_train_epochs]
        if final_train_summary:
            final_train_loss_val = final_train_summary[-1]['train_loss']
            ax.axhline(y=final_train_loss_val, color='green', linestyle=':', label=f'Final Train Loss ({final_train_loss_val:.4f})')

        ax.set_title('Training and Validation Loss Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.grid(True)
        ax.legend()
        ax.set_ylim(bottom=0)

        loss_png_file_path = os.path.join(self.progress_images_dir, f"training_loss_progress_{current_timestamp}.png")
        plt.savefig(loss_png_file_path)
        plt.close()
        logger.info(f"'{loss_png_file_path}' 파일에 학습 손실 그래프가 저장되었습니다.")


    def _plot_metrics_progress(self, eval_metrics_logs, current_timestamp):
        """
        검증 지표 (Accuracy, Precision, Recall, F1-Score) 추이를 그래프로 그리고 저장합니다.
        """
        if not eval_metrics_logs:
            logger.warning("평가 지표 데이터를 찾을 수 없어 Validation Metrics 그래프를 그릴 수 없습니다.")
            return

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        epochs_eval_metrics = [x['epoch'] for x in eval_metrics_logs]
        eval_accuracies = [x['eval_accuracy'] for x in eval_metrics_logs]
        eval_precisions = [x['eval_precision'] for x in eval_metrics_logs]
        eval_recalls = [x['eval_recall'] for x in eval_metrics_logs]
        eval_f1_scores = [x['eval_f1'] for x in eval_metrics_logs]

        ax.plot(epochs_eval_metrics, eval_accuracies, label='Validation Accuracy', color='purple', marker='o')
        ax.plot(epochs_eval_metrics, eval_precisions, label='Validation Precision', color='orange', marker='s')
        ax.plot(epochs_eval_metrics, eval_recalls, label='Validation Recall', color='green', marker='^')
        ax.plot(epochs_eval_metrics, eval_f1_scores, label='Validation F1-Score', color='brown', marker='x')

        ax.set_title('Validation Metrics Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_ylim(0.6, 1.01) # y축 범위 조정 (사용자 요청에 따라)
        ax.grid(True)
        ax.legend()

        metrics_png_file_path = os.path.join(self.progress_images_dir, f"validation_metrics_progress_{current_timestamp}.png")
        plt.savefig(metrics_png_file_path)
        plt.close()
        logger.info(f"'{metrics_png_file_path}' 파일에 검증 지표 그래프가 저장되었습니다.")


    def plot_training_progress(self):
        """
        학습 손실 및 검증 지표 그래프를 각각 다른 파일로 생성하고 저장합니다.
        """
        if self.trainer is None or not hasattr(self.trainer.state, 'log_history') or not self.trainer.state.log_history:
            logger.warning("\n학습 로그 기록을 찾을 수 없어 그래프를 그릴 수 없습니다. Trainer 설정 또는 학습 진행을 확인하세요.")
            return

        # 로그 기록에서 데이터 추출
        training_loss_logs = [x for x in self.trainer.state.log_history if 'loss' in x and 'eval_loss' not in x]
        eval_metrics_logs = [x for x in self.trainer.state.log_history if 'eval_accuracy' in x]

        if not training_loss_logs and not eval_metrics_logs:
            logger.warning("\n훈련 및 평가 로그 데이터를 찾을 수 없어 그래프를 그릴 수 없습니다.")
            return

        current_timestamp = get_timestamp()

        # 도우미 함수 호출
        self._plot_loss_progress(training_loss_logs, eval_metrics_logs, current_timestamp)
        self._plot_metrics_progress(eval_metrics_logs, current_timestamp)

