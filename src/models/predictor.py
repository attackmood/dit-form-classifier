# predictor.py

import os
import torch
import json
from typing import Union, List, Dict, Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageClassification

from src.data.preprocessor import preprocess_form_image


from src.common.common import setup_logger
logger = setup_logger(__name__)

class DiTPredictor:
    """
    학습된 DiT 모델을 사용하여 예측을 수행하는 클래스.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.id_to_label = None
        self.label_to_id = None
        self._load_model_and_processor()

    def _load_model_and_processor(self):
        """저장된 모델, 프로세서, 라벨 매핑을 로드합니다."""
        logger.info("\n--- 저장된 모델 로드 중 (예측 수행 준비) ---")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path) # 프로세서 로드
            self.model = AutoModelForImageClassification.from_pretrained(self.model_path)
            self.model.eval() # 모델을 평가 모드로 설정
            self.model.to('cpu') # 모델을 CPU로 이동

            label_mapping_path = os.path.join(self.model_path, "label_mapping.json")
            if not os.path.exists(label_mapping_path):
                raise FileNotFoundError(f"라벨 매핑 파일이 없습니다: {label_mapping_path}")

            with open(label_mapping_path, 'r') as f:
                loaded_label_mapping = json.load(f)

            self.id_to_label = {int(k): v for k, v in loaded_label_mapping['id_to_label'].items()}
            self.label_to_id = loaded_label_mapping['label_to_id']
            logger.info("모델, 프로세서, 라벨 매핑 로드 완료.")
        except Exception as e:
            logger.error(f"모델 또는 프로세서 로드 실패: {e}", exc_info=True)
            raise

    def _get_prediction_result(self, logits: torch.Tensor, confidence_threshold: float) -> Dict[str, Any]:
        """
        모델의 로짓(logits)으로부터 예측 결과를 파싱하여 딕셔너리로 반환합니다.
        이는 predict_image와 predict_batch에서 공통으로 사용될 내부 헬퍼 함수입니다.
        """
        probabilities = torch.softmax(logits, dim=-1) # 로짓을 모든 클래스 확률의 학습이 1이 되는 실제 확률 분포

        # 예측된 클래스 ID와 이름
        predicted_class_id = torch.argmax(probabilities).item()
        predicted_class_name = self.id_to_label.get(predicted_class_id, "알 수 없는 클래스 ID")

        # 모든 클래스에 대한 확률 딕셔너리 생성
        # enumerate(probabilities)는 텐서의 각 요소에 대해 (인덱스, 값) 쌍을 반환
        prob_dict = {
            self.id_to_label.get(idx, f"ID_{idx}"): round(prob.item(), 4) # id_to_label에 없는 경우 대체
            for idx, prob in enumerate(probabilities)
        }
        # 가장 높은 예측 확률 값 찾기
        max_prob_value = 0.0
        if prob_dict:
            max_prob_value = max(prob_dict.values())

        # 임계값을 기준으로 최종 예측 결과 결정
        final_predicted_label = predicted_class_name
        if max_prob_value < confidence_threshold:
            final_predicted_label = f"알 수 없음 (확률 부족: {max_prob_value:.4f} < {confidence_threshold})"

        return {
            'predicted_label': final_predicted_label,
            'probabilities': prob_dict
        }

    def predict_image(self, image_input, confidence_threshold):
        """
        단일 이미지에 대해 예측을 수행하고 임계값 기반으로 최종 레이블을 결정합니다.
        Args:
            image_input (str or PIL.Image.Image): 예측할 이미지의 파일 경로 또는 PIL Image 객체.
            confidence_threshold (float): 예측 확률 임계값.

        Returns:
            dict: 예측 결과 딕셔너리.
                  {'image_path': str or None, 'predicted_label': str, 'probabilities': dict, 'error': str(옵션)}
        """

        original_image_path = image_input if isinstance(image_input, str) else None
        result_dict: Dict[str, Any] = {'image_path': original_image_path}

        if self.processor is None or self.model is None:
            logger.error("모델 또는 프로세서가 로드되지 않았습니다. _load_model_and_processor()를 확인하세요.")
            result_dict['error'] = "모델 또는 프로세서 로드 실패"
            result_dict['predicted_label'] = "오류"  # 기본 레이블 추가
            result_dict['probabilities'] = {}  # 기본 확률 추가

            return result_dict

        try:
            preprocessed_pil = preprocess_form_image(image_input)

            if preprocessed_pil is None:
                logger.error(f"예측을 위한 이미지 전처리 실패 (원본 로드 불가): {original_image_path}. 예측을 건너뜀.")
                result_dict['error'] = "이미지 전처리 실패 또는 로드 불가"
                result_dict['predicted_label'] = "오류"
                result_dict['probabilities'] = {}
                return result_dict

            # Hugging Face Processor를 사용하여 최종 처리
            inputs = self.processor(
                images=[preprocessed_pil],  # 단일 이미지도 리스트로 감싸서 전달 (Batching 준비)
                return_tensors="pt",
                padding=True
            )

            # 예측 시에는 기울기 계산 비활성화
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[0]  # 모델이 예측한 각 클래스에 대한 원시 점수(raw scores), 아직 확률이 아님

            prediction_details = self._get_prediction_result(logits, confidence_threshold)
            result_dict.update(prediction_details)

            return result_dict

        except Exception as e:
            logger.error(f"단일 이미지 예측 중 오류 발생 ({original_image_path}): {e}", exc_info=True)
            result_dict['error'] = str(e)
            result_dict['predicted_label'] = "오류"
            result_dict['probabilities'] = {}
            return result_dict

    def predict_batch(self, image_inputs: List[Union[str, Image.Image]], confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        여러 이미지(파일 경로 또는 PIL.Image.Image 객체 리스트)에 대해 배치 예측을 수행합니다.

        Args:
            image_inputs (List[Union[str, PIL.Image.Image]]): 예측할 이미지들의 리스트.
            confidence_threshold (float): 예측 확률 임계값.

        Returns:
            List[dict]: 각 이미지에 대한 예측 결과 딕셔너리 리스트.
        """
        if not image_inputs:
            logger.warning("predict_batch에 입력된 이미지가 없습니다.")
            return []

        if self.processor is None or self.model is None:
            logger.error("모델 또는 프로세서가 로드되지 않았습니다. _load_model_and_processor()를 확인하세요.")
            # 모든 입력에 대해 오류를 반환
            return [{'image_path': (img_in if isinstance(img_in, str) else None),
                     'predicted_label': '오류', 'probabilities': {},
                     'error': "모델 또는 프로세서 로드 실패"} for img_in in image_inputs]

        processed_images = []
        original_paths = []  # 원본 경로 또는 식별자를 저장

        for i, img_input in enumerate(image_inputs):
            try:
                preprocessed_pil = preprocess_form_image(img_input)
                if preprocessed_pil is None:
                    logger.error(f"배치 예측을 위한 이미지 전처리 실패 (원본 로드 불가): {img_input}. 이 이미지는 건너뜀.")
                    processed_images.append(None)  # 유효하지 않은 이미지 표시
                else:
                    processed_images.append(preprocessed_pil)
                original_paths.append(img_input if isinstance(img_input, str) else f"InMemory_Image_{i}")
            except Exception as e:
                logger.error(f"배치 예측 중 개별 이미지 전처리 오류 ({img_input}): {e}", exc_info=True)
                processed_images.append(None)  # 오류난 이미지 표시
                original_paths.append(img_input if isinstance(img_input, str) else f"InMemory_Image_Error_{i}")

        # 유효한 이미지만 필터링
        valid_images = [img for img in processed_images if img is not None]
        if not valid_images:
            logger.warning("유효한 이미지가 없어 배치 예측을 수행할 수 없습니다.")
            # 모든 이미지에 대한 오류 결과 반환
            return [{'image_path': path, 'predicted_label': '오류', 'probabilities': {}, 'error': "이미지 전처리 실패"}
                    for path, proc_img in zip(original_paths, processed_images) if proc_img is None]

        # Hugging Face Processor를 사용하여 배치 처리
        # self.processor가 images=valid_images를 통해 여러 PIL Image 객체를 한 번에 받아 return_tensors="pt"로 변환하면,
        # 모델은 이 배치 텐서를 입력으로 받아 한 번의 순전파로 모든 이미지에 대한 로짓(logits_batch)을 계산
        inputs = self.processor(
            images=valid_images,  # 리스트 형태로 여러 이미지 전달
            return_tensors="pt",
            padding=True,  # 모든 이미지를 동일한 크기로 패딩
            truncation=True  # 필요한 경우 잘라내기
        )

        all_results: List[Dict[str, Any]] = []
        valid_image_idx = 0  # valid_images의 인덱스를 추적

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_batch = outputs.logits  # 이제 logits_batch는 [batch_size, num_labels] 형태

        for i, img_input_orig in enumerate(image_inputs):
            current_image_path = img_input_orig if isinstance(img_input_orig, str) else None

            if processed_images[i] is None:  # 전처리 과정에서 실패한 이미지
                all_results.append({
                    'image_path': current_image_path if current_image_path else original_paths[i],
                    'predicted_label': '오류',
                    'probabilities': {},
                    'error': "이미지 전처리 실패 또는 로드 불가"
                })
            else:
                # 해당 이미지에 대한 로짓 추출
                # valid_images의 순서와 original_paths의 순서가 일치한다고 가정
                current_logits = logits_batch[valid_image_idx]
                prediction_details = self._get_prediction_result(current_logits, confidence_threshold)

                # 결과 딕셔너리 구성
                result_item = {
                    'image_path': current_image_path,
                }
                result_item.update(prediction_details)
                all_results.append(result_item)
                valid_image_idx += 1  # 다음 유효 이미지로 인덱스 이동

        return all_results
