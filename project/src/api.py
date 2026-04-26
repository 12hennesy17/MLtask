import io
import os
import sys
import logging
from time import perf_counter
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, model_validator

# --- Настройка логирования ---
from .logger_setup import setup_logging

# --- Локальные ML-модули ---
from .config import *
from .transformers import CombinedFeaturesAdded 
from .core import compute_quality_flags, missing_table, summarize_dataset


setup_logging("logs/api.log")
logger = logging.getLogger("src.api")

# Глобальный словарь для хранения загруженной модели
ml_models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения: загрузка ML-модели при старте и очистка при остановке."""
    print("⏳ Загрузка модели...")
    try:
        ml_models["pipeline"] = joblib.load(os.path.join(PRODUCTION_MODELS_DIR, "final_production_model.joblib"))
        logger.info("✅ Модель успешно загружена и готова к работе!")
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка загрузки модели: {e}", exc_info=True)     
    yield  # Сервер работает
    
    # Очистка при выключении сервера
    ml_models.clear()
    logger.info("🧹 Память очищена.")


app = FastAPI(
    title="Ames Housing Predictor API",
    description="API для предсказания цен на недвижимость (StackingRegressor) и проверки качества данных.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


# --- Схемы данных (Pydantic Models) ---

class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""
    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(..., ge=0.0, le=1.0, description="Максимальная доля пропусков (0..1)")
    numeric_cols: int = Field(..., ge=0, description="Количество числовых колонок")
    categorical_cols: int = Field(..., ge=0, description="Количество категориальных колонок")

    @model_validator(mode='after')
    def check_column_sum(self):
        if self.n_cols != (self.numeric_cols + self.categorical_cols):
            raise ValueError(
                f"Сумма колонок ({self.numeric_cols} + {self.categorical_cols}) "
                f"не совпадает с общим числом {self.n_cols}"
            )
        return self

class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""
    ok_for_model: bool = Field(..., description="True, если датасет качественный для обучения")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Интегральная оценка качества (0..1)")
    message: str = Field(..., description="Человекочитаемое пояснение решения")
    latency_ms: float = Field(..., ge=0.0, description="Время обработки запроса, мс")
    flags: dict[str, bool] | None = Field(default=None, description="Булевы флаги (too_few_rows и т.д.)")
    dataset_shape: dict[str, int] | None = Field(default=None, description="Размеры датасета")
# --- Эндпоинт: Системный ---

@app.get("/health", tags=["System"])
def healthcheck() -> dict[str, Any]:
    """Комплексная проверка: работает ли API и загружена ли ML-модель."""
    model_loaded = "pipeline" in ml_models
    
    if not model_loaded:
        logger.warning("⚠️ Запрос /health: сервис запущен, но ML-модель отсутствует в памяти.")
        raise HTTPException(
            status_code=503, 
            detail="Сервис запущен, но ML-модель не загружена."
        )
    
    return {
        "status": "ok",
        "service": "ames-housing-ml-api",
        "version": "1.0.0",
        "model_ready": True
    }


# --- Эндпоинт: Инференс (ML Predictions) ---

@app.post("/predict/csv", tags=["Inference"])
async def predict_from_csv(file: UploadFile = File(...)):
    """Принимает CSV файл, делает предсказания и возвращает JSON"""
    logger.info(f"📥 Получен запрос на предсказание. Файл: {file.filename}")
    if not file.filename.endswith('.csv'):
        logger.warning(f"⚠️ Отклонен файл с неверным расширением: {file.filename}")
        raise HTTPException(status_code=400, detail="Ожидается файл формата .csv")
    
    try:
        # Читаем файл из памяти
        start_time = perf_counter()
        await file.seek(0)
        df = await run_in_threadpool(pd.read_csv, file.file)
        logger.info(f"📊 Файл прочитан. Строк: {df.shape[0]}, Колонок: {df.shape[1]}")

        # Очистка
        X_test = df.drop(columns=[TARGET, 'Id'], errors='ignore')
        
        # Предсказание
        preds_log = await run_in_threadpool(ml_models["pipeline"].predict, X_test)
        
        # Обратное преобразование (из логарифма в реальные цены)
        preds_actual = await run_in_threadpool(np.expm1, preds_log)
        
        # Формируем ответ
        result = [
            {"Id": int(id_val) if 'Id' in df.columns else idx, "SalePrice": float(price)}
            for idx, (id_val, price) in enumerate(zip(df.get('Id', df.index), preds_actual))
        ]
        latency = (perf_counter() - start_time) * 1000
        logger.info(f"✅ Предсказание выполнено для {len(result)} строк за {latency:.1f} мс")
        return {"predictions": result}
        
    except Exception as e:
        logger.error(f"❌ Ошибка при генерации предсказаний для файла {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


# --- Эндпоинт: Качество данных (EDA / Quality) ---

@app.post("/quality", response_model=QualityResponse, tags=["Quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """Оценка качества датасета на основе переданных агрегированных метрик."""
    start = perf_counter()

    score = 1.0
    score -= req.max_missing_share / req.n_cols
    if req.n_rows < 1000: score -= 0.2
    if req.n_cols > 100: score -= 0.1
    if req.numeric_cols == 0 and req.categorical_cols > 0: score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0: score -= 0.05

    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    
    message = (
        "Данных достаточно, модель можно обучать." 
        if ok_for_model 
        else "Качество данных недостаточно, требуется доработка."
    )

    latency_ms = (perf_counter() - start) * 1000.0

    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    logger.info(
        f"🔍 [quality] Запрос: n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing={req.max_missing_share:.3f} | "
        f"Result: score={score:.3f} ok={ok_for_model} latency={latency_ms:.1f}ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["Quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Принимает CSV-файл и возвращает оценку качества данных.
    """
    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        logger.warning(f"⚠️ Неверный content-type: {file.content_type} для файла {file.filename}")
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # "Перематываем" файл в начало. 
        # Это важно, если до этого другие функции или проверки уже читали файл.
        await file.seek(0)
        df = await run_in_threadpool(pd.read_csv, file.file)
    except Exception as exc:
        logger.error(f"❌ Ошибка чтения CSV файла {file.filename} в эндпоинте quality: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        logger.warning(f"⚠️ Получен пустой DataFrame из файла {file.filename}")
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    message = (
        "CSV выглядит достаточно качественным для обучения модели." 
        if ok_for_model 
        else "CSV требует доработки перед обучением модели."
    )

    latency_ms = (perf_counter() - start) * 1000.0

    flags_bool: dict[str, bool] = {
        key: bool(value) for key, value in flags_all.items() if isinstance(value, bool)
    }

    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    logger.info(
        f"Processed CSV: filename={file.filename!r} | "
        f"dims=({n_rows}x{n_cols}) | score={score:.3f} | "
        f"latency={latency_ms:.1f}ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )

if __name__ == "__main__":
    import uvicorn
    from src.config import API_PORT, API_RELOAD
    # Тут он сам подтянет reload из кода или env
    uvicorn.run("src.api:app", host="0.0.0.0", port=API_PORT, reload=API_RELOAD)