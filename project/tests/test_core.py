import pytest
import pandas as pd
import numpy as np

# Импортируем тестируемые функции
from src.core import (
    missing_table,
    clean_outliers,
    log_transform_target,
    summarize_dataset,
    compute_metrics,
    get_low_variance_report,
    check_categorical_redundancy,
    correlation_matrix
)

@pytest.fixture
def dummy_df():
    """Создаем игрушечный датафрейм со всеми 'проблемами', которые хотим поймать."""
    return pd.DataFrame({
        'Id': [1, 2, 3, 4, 5],
        'Price': [100, 200, 300, 400, 500],
        'WithMissing': [1.0, np.nan, 3.0, np.nan, 5.0],
        'ConstantCol': ['A', 'A', 'A', 'A', 'A'],
        'RedundantCol': ['A', 'A', 'A', 'A', 'A']
    })

def test_missing_table(dummy_df):
    """Проверяем правильность расчета пропусков."""
    res = missing_table(dummy_df)
    
    # В таблице пропусков должна быть только одна колонка (остальные без пропусков)
    assert len(res) == 1
    assert res.index[0] == 'WithMissing'
    
    # 2 пропуска из 5 строк = 40% (0.4)
    assert res.loc['WithMissing', 'missing_count'] == 2
    assert res.loc['WithMissing', 'missing_share'] == 0.4

def test_clean_outliers(dummy_df):
    """Проверяем удаление выбросов по списку ID."""
    outlier_ids = [2, 4]
    cleaned = clean_outliers(dummy_df, outlier_ids)
    
    assert len(cleaned) == 3 # Осталось 3 строки
    assert 2 not in cleaned['Id'].values
    assert 4 not in cleaned['Id'].values
    assert 1 in cleaned['Id'].values # Нормальные строки остались

def test_log_transform_target():
    """Проверяем, что используется именно np.log1p (логарифм от x+1)."""
    y = pd.Series([0, np.exp(1) - 1])
    transformed = log_transform_target(y)
    
    assert np.isclose(transformed[0], 0.0)
    assert np.isclose(transformed[1], 1.0)

def test_summarize_dataset(dummy_df):
    """Проверяем, что summary собирает правильные метаданные."""
    summary = summarize_dataset(dummy_df)
    
    assert summary.n_rows == 5
    assert summary.n_cols == 5
    
    # Проверяем конкретную колонку
    price_col = next(c for c in summary.columns if c.name == 'Price')
    assert price_col.is_numeric is True
    assert price_col.min == 100
    assert price_col.max == 500
    assert price_col.missing == 0
    
    const_col = next(c for c in summary.columns if c.name == 'ConstantCol')
    assert const_col.is_numeric is False
    assert const_col.unique == 1

def test_get_low_variance_report(dummy_df):
    """Проверяем поиск константных признаков."""
    report = get_low_variance_report(dummy_df, threshold=0.9)
    
    # ConstantCol и RedundantCol имеют 100% одинаковых значений
    assert 'ConstantCol' in report['feature'].values
    assert 'RedundantCol' in report['feature'].values
    
    # Price имеет 5 разных значений, его там быть не должно
    assert 'Price' not in report['feature'].values

def test_check_categorical_redundancy(dummy_df):
    """Проверяем поиск дублирующихся категориальных колонок."""
    report = check_categorical_redundancy(dummy_df, threshold=0.9)
    
    assert not report.empty
    # ConstantCol и RedundantCol совпадают на 100%
    assert report.iloc[0]['feat1'] in ['ConstantCol', 'RedundantCol']
    assert report.iloc[0]['feat2'] in ['ConstantCol', 'RedundantCol']
    assert report.iloc[0]['match_ratio'] == 1.0

def test_compute_metrics():
    """Проверка математики метрик регрессии."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5,  0.0, 2.0, 8.0])
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'R2' in metrics
    
    # MAE = (|0.5| + |0.5| + |0| + |-1.0|) / 4 = 2.0 / 4 = 0.5
    assert metrics['MAE'] == 0.5

def test_correlation_matrix(dummy_df):
    """Корреляционная матрица должна игнорировать текст и считать только числа."""
    corr = correlation_matrix(dummy_df)
    
    # Строковые колонки 'ConstantCol' и 'RedundantCol' должны быть исключены
    assert 'ConstantCol' not in corr.columns
    assert 'Id' in corr.columns
    assert 'Price' in corr.columns
    
    # Корреляция переменной самой с собой равна 1.0
    assert corr.loc['Price', 'Price'] == 1.0