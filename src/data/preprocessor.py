# preprocessor.py

import cv2
import numpy as np
from PIL import Image

from src.common.common import setup_logger

# 로거 설정
logger = setup_logger(__name__)

def preprocess_form_image(image_input):
    """
    검사지 양식 분류를 위한 핵심 전처리 함수 (기울기 보정만).
    기울기 보정 후 PIL Image 객체를 반환합니다.
    """
    try:
        # 1. 이미지를 RGB 3채널 형식 PIL Image 객체로 변환
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert('RGB')
        else:
            raise TypeError("image_input은 PIL.Image 객체 또는 파일 경로(str)여야 합니다.")

        img_cv_color = np.array(img_pil) # OpenCV 처리를 위해 NumPy 배열 (H, W, C)

        # 2. 기울기 보정 (Deskewing) - 흑백 이미지로 각도 감지 후 컬러 이미지 회전
        img_gray_for_skew = cv2.cvtColor(img_cv_color, cv2.COLOR_RGB2GRAY)

        _, binary_for_skew = cv2.threshold(img_gray_for_skew, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(binary_for_skew > 0))
        if len(coords) < 100: # 텍스트가 너무 적으면 기울기 감지 어려움
            logger.warning("기울기 감지를 위한 픽셀이 너무 적습니다. 기울기 보정을 건너뛰고 원본 컬러 이미지를 반환합니다.")
            return img_pil # PIL Image 그대로 반환 (Processor가 처리)

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img_gray_for_skew.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_deskewed_color_np = cv2.warpAffine(
            img_cv_color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        # NumPy 배열을 PIL Image로 다시 변환하여 반환
        return Image.fromarray(img_deskewed_color_np.astype(np.uint8), 'RGB')


    except Exception as e:
        logger.error(f"이미지 전처리 중 오류 발생: {e}. 원본 PIL Image를 반환합니다.")
        if isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        else:
            # 파일 경로가 전달되었을 때, 오류 시에도 PIL Image를 반환하도록 시도
            try:
                return Image.open(image_input).convert('RGB')
            except Exception as fe:
                logger.error(f"오류 처리 중에도 이미지 로드 실패: {fe}. None 반환.")
                return None # 최후의 수단, None을 반환하여 이후 처리에서 오류 발생하도록 유도