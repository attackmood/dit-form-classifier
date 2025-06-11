# src/services/predict_service.py

from io import BytesIO
from typing import List, Dict, Any, Union, Tuple

from PIL import Image

from config import config
from src.common.common import setup_logger
from src.models.predictor import DiTPredictor

logger = setup_logger(__name__)

# DiTPredictor 인스턴스를 저장할 전역 변수 (초기에는 None)
_predictor_instance: DiTPredictor = None


def get_predictor() -> DiTPredictor:
    """
    서비스 레이어에서 사용할 DiTPredictor 인스턴스를 반환합니다.
    인스턴스가 없으면 새로 생성하여 로드합니다.
    """
    global _predictor_instance
    logger.info("PredictService: DiTPredictor 인스턴스 로드 중...")
    if _predictor_instance is None:
        try:
            _predictor_instance = DiTPredictor(config.FINE_TUNED_MODEL_PATH)
            logger.info("PredictService: DiTPredictor 인스턴스 로드 완료.")
        except Exception as e:
            logger.error(f"PredictService: DiTPredictor 초기화 실패: {e}", exc_info=True)
            raise RuntimeError(f"모델 초기화 실패: {e}")
    return _predictor_instance


def _convert_bytes_to_pil_images(file_contents_list: List[bytes], filenames: List[str]) -> Tuple[
    List[Union[Image.Image, None]], List[str]]:
    """
    바이트 데이터 리스트를 PIL Image 객체 리스트로 변환합니다.
    변환에 실패한 이미지는 None으로 표시하고, 해당 원본 식별자를 반환합니다.

    Returns:
        Tuple[List[Union[PIL.Image.Image, None]], List[str]]:
        변환된 PIL Image 리스트 (실패 시 None 포함)와 해당 이미지들의 원본 식별자 리스트.
    """
    pil_images: List[Union[Image.Image, None]] = []
    original_identifiers: List[str] = []

    # 각 바이트 데이터를 PIL Image 객체로 변환
    for i, content in enumerate(file_contents_list):
        # 파일명 리스트의 길이를 고려하여 식별자 생성
        identifier = filenames[i] if i < len(filenames) else f"upload_{i}"
        try:
            img_pil = Image.open(BytesIO(content)).convert("RGB")
            pil_images.append(img_pil)
            original_identifiers.append(identifier)
        except Exception as e:
            logger.error(f"Service: 다중 이미지 중 '{identifier}' 변환 실패: {e}", exc_info=True)
            pil_images.append(None)  # 변환 실패 표시
            original_identifiers.append(identifier)  # 실패한 경우에도 식별자는 유지

    return pil_images, original_identifiers


class PredictService:
    """
    FastAPI 요청을 처리하기 위한 비즈니스 로직을 담는 서비스 클래스.
    DiTPredictor를 활용합니다.
    """

    def __init__(self):
        self.predictor = get_predictor()

    async def predict_single_image_from_upload(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        업로드된 단일 이미지 파일의 바이트 데이터를 받아 예측을 수행합니다.
        임시 파일 저장 없이 메모리 내에서 처리합니다.
        """
        logger.info(f"Service: 단일 이미지 예측 요청 처리 중: {filename}")
        try:
            # BytesIO를 사용하여 메모리에서 이미지 열기
            image_pil = Image.open(BytesIO(file_content)).convert("RGB")

            # 단일 이미지 예측 함수 호출
            prediction_result = self.predictor.predict_image(image_pil, config.THRESHOLD)

            # 결과에 원본 파일 이름을 추가 (필요시)
            if prediction_result:  # prediction_result가 유효한 딕셔너리인지 먼저 확인
                current_image_path = prediction_result.get('image_path')  # 키가 없으면 None 반환

                if current_image_path is None or str(current_image_path).startswith('InMemory_Image'):
                    prediction_result['image_path'] = filename

            return prediction_result

        except Exception as e:
            logger.error(f"Service: 단일 이미지({filename}) 예측 중 오류 발생: {e}", exc_info=True)
            return {"image_path": filename, "predicted_label": "오류", "probabilities": {}, "error": str(e)}

    async def predict_multiple_images_from_uploads(self, file_contents_list: List[bytes], filenames: List[str]) -> List[
        Dict[str, Any]]:
        logger.info(f"Service: 다중 이미지 예측 요청 처리 중. 파일 수: {len(filenames)}")
        if not file_contents_list:
            return []

        # 이미지 변환 및 식별자 준비
        pil_images, original_identifiers = _convert_bytes_to_pil_images(file_contents_list, filenames)
        # 이미지 예측 배치 처리 함수 호출
        batch_prediction_results = self.predictor.predict_batch(pil_images, config.THRESHOLD)

        final_results = []
        for i, res in enumerate(batch_prediction_results):
            # predict_batch의 image_path가 None 또는 InMemory_Image_X 일 경우, 업로드된 파일명으로 대체
            # original_identifiers는 pil_images와 동일한 인덱스를 가집니다.
            if res.get('image_path') is None or str(res.get('image_path', '')).startswith('InMemory_Image'):
                res['image_path'] = original_identifiers[i]
            final_results.append(res)

        return final_results
