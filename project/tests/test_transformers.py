import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.config import *
from src.transformers import MissingValueImputer, CombinedFeaturesAdded, OrdinalEncoderTransformer
from src.pipeline import get_pipeline
import yaml 
import os

@pytest.fixture
def sample_data():
    """Эталонный набор данных для тестов."""
    df = pd.read_csv(TRAIN_DATA)
    return df.head()

def test_missing_value_imputer(sample_data):
    imputer = MissingValueImputer()
    transformed = imputer.fit_transform(sample_data)
    assert transformed.isnull().sum().sum() == 0, "Остались пропуски после Imputer"

def test_combined_features_added(sample_data):
    # 1. Подготовка
    data = MissingValueImputer().fit_transform(sample_data)
    fe = CombinedFeaturesAdded(drop_originals=True)
    transformed = fe.fit_transform(data)
    
    # 2. Проверка наличия ожидаемых колонок
    expected_cols = ['TotalSF', 'HouseAge', 'IsNearRoad', 'LivingArea_per_Bedroom']
    for col in expected_cols:
        assert col in transformed.columns, f"Колонка {col} отсутствует после генерации признаков"
    
    # 3. Проверка типов данных (dtypes)
    assert pd.api.types.is_numeric_dtype(transformed['TotalSF']), "TotalSF должен быть числом"
    assert transformed['IsNearRoad'].isin([0, 1]).all(), "IsNearRoad должен быть бинарным (0 или 1)"
    
    # 4. Проверка диапазонов (Logic-based constraints)
    assert (transformed['HouseAge'] >= 0).all(), "Возраст дома не может быть отрицательным"
    
    # Проверяем, что площадь больше 0
    assert (transformed['TotalSF'] > 0).all(), "Площадь должна быть положительной"
    
    # Проверка, что после drop_originals=True исходных колонок нет
    assert 'GrLivArea' not in transformed.columns, "Исходная колонка GrLivArea не была удалена"

def test_ordinal_encoder():
    """Проверяем, что энкодер корректно читает конфиг и маппит значения."""

    with open(os.path.join(TRAIN_PARAMS_DIR, 'data_params.yaml'), 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
            
    # Берем маппинг для одной колонки как пример
    mapping = params['ordinal_mappings']['ExterQual']
    
    # Создаем маленький, контролируемый DataFrame
    df = pd.DataFrame({'ExterQual': ['Ex', 'Gd', 'TA', 'Fa']})
    
    # Инициализируем и трансформируем
    encoder = OrdinalEncoderTransformer(column_mappings={'ExterQual': mapping})
    transformed = encoder.fit_transform(df)
    
    # Сверяем результат с логикой из YAML
    assert transformed['ExterQual'].iloc[0] == 5  # Ex
    assert transformed['ExterQual'].iloc[1] == 4  # Gd
    assert transformed['ExterQual'].iloc[2] == 3  # TA
    assert transformed['ExterQual'].dtype == int  # Убеждаемся, что тип числовой

def test_pipeline_execution(sample_data):
    """Тестируем весь путь данных через пайплайн."""
    pipeline = get_pipeline()
    
    output = pipeline.fit_transform(sample_data)

    assert output.shape[1] > 0, "Пайплайн вернул пустой результат"
    assert not np.isnan(output).any(), "В данных остались NaN после пайплайна"