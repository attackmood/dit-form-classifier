from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from src.common.common import setup_logger
from src.services.predict_service import PredictService

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 시작 및 종료 시점을 관리합니다.
    서버 시작 시 PredictService 인스턴스를 초기화하고 app.state에 저장합니다.
    """
    logger.info("애플리케이션 시작: PredictService 초기화 중...")
    try:
        app.state.predict_service = PredictService()
        logger.info("애플리케이션 시작: PredictService 인스턴스 생성 완료.")

        yield  # 애플리케이션이 실행되는 동안 대기

    except Exception as e:
        logger.error(f"애플리케이션 시작 실패: {e}", exc_info=True)
    finally:
        logger.info("애플리케이션 종료.")


# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(lifespan=lifespan, title="API입니당")


@app.get("/")
async def root():
    return {"message": "DiT Form Classifier API is running!"}


@app.post("/predict/single_image")
async def predict_single_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    단일 이미지 파일을 업로드하여 양식을 분류합니다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{file.filename}' 파일은 이미지 형식이 아닙니다.")

    try:
        file_content = await file.read()
        filename = file.filename if file.filename else "unknown_image"

        predict_service: PredictService = app.state.predict_service

        if predict_service is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="예측 서비스가 초기화되지 않았습니다.")

        # PredictionService의 단일 이미지 예측 메소드 호출
        result = await predict_service.predict_single_image_from_upload(file_content, filename)

        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"단일 이미지 예측 API 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"예측 처리 중 예상치 못한 오류 발생: {e}"
        )


@app.post("/predict/multiple_images")
async def predict_multiple_images(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    여러 이미지 파일을 업로드하여 양식을 분류합니다.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="하나 이상의 이미지 파일을 업로드해야 합니다.")

    file_contents_list = []
    filenames = []

    # 각 파일의 유효성 검사 및 내용 읽기
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{file.filename}' 파일은 이미지 형식이 아닙니다.")
        file_contents_list.append(await file.read())
        filenames.append(file.filename if file.filename else "unknown_image")

    try:
        predict_service: PredictService = app.state.predict_service

        if predict_service is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="예측 서비스가 초기화되지 않았습니다.")

        # PredictionService의 다중 이미지 예측 메소드 호출
        results = await predict_service.predict_multiple_images_from_uploads(file_contents_list, filenames)

        return JSONResponse(content=results, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"다중 이미지 예측 API 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"예측 처리 중 예상치 못한 오류 발생: {e}"
        )
