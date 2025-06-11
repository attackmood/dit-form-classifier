# predict.py

import json
import os
import sys
from typing import List, Dict, Any

from src.common.common import get_timestamp, setup_logger
from config import config
from src.models.predictor import DiTPredictor

# 로거 설정
logger = setup_logger(__name__)

# --------------------------------
# 전역 DiTModelPredictor 인스턴스 (한 번만 로드)
_predictor_instance = None # 초기에는 None으로 설정

def get_global_predictor():
    """
    DiTModelPredictor 인스턴스를 반환합니다.
    없으면 초기화하고 반환합니다. (싱글톤 패턴)
    """
    global _predictor_instance
    if _predictor_instance is None:
        try:
            logger.info("DiTModelPredictor 초기화 중...")
            _predictor_instance = DiTPredictor(model_path=config.SAVE_MODEL_PATH)
            logger.info("DiTModelPredictor 초기화 완료.")
        except Exception as e:
            logger.error(f"DiTModelPredictor 초기화 중 오류 발생: {e}", exc_info=True)
            raise RuntimeError(f"모델 초기화 실패: {e}")
    return _predictor_instance



def run_prediction_workflow_single(image_path: str, threshold: float) -> Dict[str, Any]:
    """
    단일 이미지에 대한 예측 워크플로우를 실행하고 결과를 반환합니다.
    """
    logger.info(f"--- 단일 이미지 예측 워크플로우 시작: {image_path} ---")
    predictor = get_global_predictor()

    if not os.path.exists(image_path):
        logger.error(f"지정된 이미지 파일이 존재하지 않습니다: {image_path}")
        return {"image_path": image_path, "error": "파일을 찾을 수 없습니다."}

    try:
        # DiTModelPredictor의 predict_image 메서드 호출
        prediction_result = predictor.predict_image(image_path, threshold)

        if 'error' in prediction_result:
            logger.error(f"이미지 예측 실패: {prediction_result.get('error')}")
        else:
            prob_display = json.dumps(prediction_result.get('probabilities', {}), ensure_ascii=False)
            logger.info(
                f"이미지: {os.path.basename(image_path)}, "
                f"예측: {prediction_result.get('predicted_label', 'N/A')}, "
                f"확률: {prob_display}"
            )
        return prediction_result

    except Exception as e:
        logger.error(f"단일 이미지 예측 워크플로우 실행 중 오류 발생 ({image_path}): {e}", exc_info=True)
        return {"image_path": image_path, "error": str(e)}
    finally:
        logger.info(f"--- 단일 이미지 예측 워크플로우 완료: {image_path} ---")


def run_prediction_workflow_directory(directory_path: str, threshold: float) -> List[Dict[str, Any]]:
    """
    지정된 디렉토리 내 모든 이미지에 대한 예측 워크플로우를 실행하고 결과를 반환합니다.
    """
    logger.info(f"--- 디렉토리 예측 워크플로우 시작: {directory_path} ---")
    predictor = get_global_predictor() # 전역 predictor 인스턴스 사용
    all_prediction_results: List[Dict[str, Any]] = []

    try:
        # 1. 검증 폴더에서 이미지 파일 목록 가져오기 (os.walk 사용)
        if not os.path.isdir(directory_path):
            logger.error(f"오류: 지정된 디렉토리 '{directory_path}'를 찾을 수 없습니다.")
            return []

        image_files_to_predict = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files_to_predict.append(os.path.join(root, file))

        image_files_to_predict.sort() # 일관된 순서를 위해 정렬

        if not image_files_to_predict:
            logger.warning(f"'{directory_path}' 디렉토리에서 예측할 이미지 파일을 찾을 수 없습니다. 지원되는 형식: {image_extensions}")
            return []

        logger.info(f"총 {len(image_files_to_predict)}개의 이미지를 찾았습니다. 배치 예측을 시작합니다.")

        # DiTPredictor의 predict_batch 메소드를 사용
        batch_results = predictor.predict_batch(image_files_to_predict, threshold)
        all_prediction_results.extend(batch_results)

        # 각 결과에 대한 로깅
        for result_item in all_prediction_results:
            img_path = result_item.get('image_path', '알 수 없는 파일')
            if 'error' in result_item:
                logger.error(f"'{os.path.basename(img_path)}' 예측 실패: {result_item['error']}", exc_info=False)
            else:
                prob_display = json.dumps(result_item.get('probabilities', {}), ensure_ascii=False)
                logger.info(f"이미지: {os.path.basename(img_path)}, 예측: {result_item.get('predicted_label', 'N/A')}, 확률: {prob_display}")

        return all_prediction_results

    except Exception as e:
        logger.error(f"디렉토리 예측 중 오류 발생: {e}", exc_info=True)
        return [] # 오류 발생 시 현재까지의 결과 반환 또는 빈 리스트
    finally:
        logger.info("--- 디렉토리 예측 워크플로우 완료 ---")


def save_result(predictions: list):
    """
    예측 결과를 JSON 파일로 저장합니다.
    """
    if not predictions:
        logger.warning("저장할 예측 결과가 없습니다.")
        return

    output_dir_path = config.PRED_SAVE_PATH
    # 파일 이름에 타임스탬프 추가
    timestamp = get_timestamp()
    output_filename = f"image_predictions_{timestamp}.json"

    # Path 객체 연산자를 사용하여 최종 파일 경로 구성
    output_file_path = output_dir_path / output_filename

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        logger.info(f"디렉토리 예측 결과를 '{output_file_path}'에 성공적으로 저장했습니다.")
    except Exception as e:
        logger.error(f"예측 결과 저장 중 오류 발생: {e}", exc_info=True)


if __name__ == '__main__':
    test_single_image_path = "F:\\Datasets\\test_dataset\\신규더맘의뢰서.jpg" # 실제 테스트 이미지 경로로 변경 필요
    test_directory_path =  "F:\\Datasets\\request_form\\model_test_data"

    logger.info("예측 스크립트 시작.")

    # 모델 미리 로드
    try:
        get_global_predictor()
        logger.info("모델 로드 완료.")
    except RuntimeError:
        sys.exit(1) # 모델 로드 실패 시 종료

    # 단일 이미지 예측 테스트
    # single_prediction_result = run_prediction_workflow_single(test_single_image_path, config.THRESHOLD)
    # if single_prediction_result:
    #     save_result([single_prediction_result])  # 단일 결과도 리스트로 만들어 저장

    # 디렉토리 예측 테스트
    directory_predictions = run_prediction_workflow_directory(test_directory_path, config.THRESHOLD)
    if directory_predictions:
        save_result(directory_predictions)

    logger.info("예측 스크립트 종료.")