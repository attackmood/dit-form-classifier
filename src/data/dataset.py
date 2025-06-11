# dataset.py

import os

import torch
from datasets import Dataset, Image as HF_Image
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor

from config import config
from src.common.common import setup_logger
from .preprocessor import preprocess_form_image

logger = setup_logger(__name__)

def create_dit_dataset():
    """
    데이터셋을 로드하고, 분할하며, Hugging Face Dataset 형식으로 전처리합니다.
    """
    # 1. 이미지 경로와 라벨 수집
    image_paths = []
    labels = []
    label_to_id = {name: i for i, name in enumerate(config.CLASS_NAMES)} # 라벨 이름을 숫자 ID로 매핑
    id_to_label = {i: name for i, name in enumerate(config.CLASS_NAMES)} # 숫자 ID를 라벨 이름으로 매핑

    logger.info("\n--- 1단계: 데이터셋 준비 시작 ---")
    for class_name in config.CLASS_NAMES:
        class_path = os.path.join(config.DATASET_BASE_DIR, class_name)
        if not os.path.isdir(class_path):
            logger.warning(f"폴더 '{class_path}'를 찾을 수 없습니다. '{class_name}' 클래스는 건너뜁니다.")
            continue

        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(class_path, img_name))
                labels.append(label_to_id[class_name]) # <-- 여기에 숫자로 매핑된 라벨이 추가됩니다.

    if not image_paths:
        logger.error("데이터셋 폴더에서 이미지를 찾을 수 없습니다. 경로와 파일 형식을 확인해 주세요.")
        return None, None, label_to_id, id_to_label, None

    logger.info(f"총 {len(image_paths)}개의 이미지와 라벨을 수집했습니다.")
    logger.info(f"라벨 매핑: {label_to_id}")

    # 2. 훈련/검증/테스트 세트 분리
    # sklearn 패키지의 train_test_split : 주어진 데이터셋을 훈련세트와 테스트세트로 분할해줌(과적합 문제 방지)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=labels
    )

    logger.info(f"훈련 이미지 수: {len(train_paths)}, 테스트 이미지 수: {len(test_paths)}")

    # 3. Hugging Face Dataset 객체 생성
    train_dict = {"image": train_paths, "label": train_labels}
    test_dict = {"image": test_paths, "label": test_labels}

    # dict 타입을 받아서 Dataset 객체로 변환(image 컬럼 데이터 타입을 파일경로에서 Image() 타입으로 변경)
    train_dataset_raw = Dataset.from_dict(train_dict).cast_column("image", HF_Image())
    test_dataset_raw = Dataset.from_dict(test_dict).cast_column("image", HF_Image())
    logger.info("\n--- 1단계: 데이터셋 준비 완료 ---")


    # 4. Hugging Face Processor 로드
    processor = AutoProcessor.from_pretrained(config.MODEL_NAME)
    logger.info("DiT 프로세서 로드 완료.")

    def _preprocess_data(examples):
        # 1. 이미지 전처리(왜곡 보정) 함수로 이미지 전처리 후 이미지로 반환
        # 2. 전처리된 이미지들을 processor를 통해 전처리(리사이징, 정규화 등) 후 모델에 입력할 수 있는 PyTorch 텐서 형식으로 변환
        # 3. 라벨도 PyTorch 텐서로 변환
        processed_pil_images = [preprocess_form_image(img_pil_obj) for img_pil_obj in examples['image']]
        inputs = processor(images=processed_pil_images, return_tensors="pt", padding=True)
        inputs["labels"] = torch.tensor(examples['label'], dtype=torch.long)
        return inputs


    logger.info("\n--- 2단계: 데이터셋 전처리 및 Hugging Face Dataset 생성 중... (시간이 걸릴 수 있습니다) ---")

    # 5. 데이터셋의 각 데이터 항목에 _preprocess_data 함수를 적용하여 이미지를 전처리하고 DiT 모델 학습용 텐서로 변환
    train_dataset = train_dataset_raw.map(_preprocess_data, batched=True, remove_columns=['image'], num_proc=1)
    test_dataset = test_dataset_raw.map(_preprocess_data, batched=True, remove_columns=['image'], num_proc=1)

    # 데이터셋 포맷을 PyTorch 텐서로 설정 (datasets 라이브러리가 데이터를 PyTorch 모델이 바로 인식할 수 있는 텐서 형태로 변환해주도록 지시)
    train_dataset.set_format(type="torch", columns=['pixel_values', 'labels'])
    test_dataset.set_format(type="torch", columns=['pixel_values', 'labels'])

    logger.info("\n--- 2단계: 데이터셋 전처리 및 Hugging Face Dataset 생성 완료 ---")
    logger.info(f"훈련 데이터셋 첫 번째 샘플 형태: {train_dataset[0]['pixel_values'].shape}")
    logger.info(f"훈련 데이터셋 첫 번째 라벨: {train_dataset[0]['labels'].item()}")

    return train_dataset, test_dataset, label_to_id, id_to_label, processor