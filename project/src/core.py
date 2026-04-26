# -*- coding: utf-8 -*-
from __future__ import annotations

# --- Стандартные библиотеки ---
import os
import sys
import logging
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# --- Математика и анализ данных ---
import numpy as np
import pandas as pd
import scipy.stats as stats
from pandas.api import types as ptypes
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Scikit-learn: Метрики и Валидация ---
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# --- Scikit-learn: Интерпретируемость и Фичи ---
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.feature_selection import mutual_info_regression

# --- Вспомогательные инструменты ---
import joblib

# --- Локальные импорты ---
from .config import FOLDERS_TO_CREATE

@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: list[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: list[ColumnSummary]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: list[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    result = result[result['missing_count'] > 0].sort_values(by='missing_share', ascending=False)
    return result

def clean_outliers(df: pd.DataFrame, outlier_ids: list[int]) -> pd.DataFrame:
    """Удаление специфических ID, найденных в ходе EDA."""
    return df[~df['Id'].isin(outlier_ids)].copy()

def log_transform_target(y: pd.Series) -> pd.Series:
    """Логарифмирование таргета (SalePrice)."""
    return np.log1p(y)

def get_low_variance_report(df: pd.DataFrame, threshold: float = 0.93) -> pd.DataFrame:
    """
    Анализирует признаки с низкой вариативностью.
    Возвращает DataFrame с метриками для колонок, где доминирующее значение > threshold.
    """
    report_data = []
    
    for col in df.columns:
        # Считаем доли каждого значения
        value_counts_norm = df[col].value_counts(normalize=True)
        
        if not value_counts_norm.empty:
            most_freq_share = value_counts_norm.iloc[0] # Доля самого частого
            most_freq_value = value_counts_norm.index[0] # Само значение
            
            if most_freq_share > threshold:
                report_data.append({
                    'feature': col,
                    'dominant_value': most_freq_value,
                    'share_%': round(most_freq_share * 100, 2),
                    'unique_count': df[col].nunique()
                })
    
    # Создаем табличку и сортируем по убыванию доминирования
    report_df = pd.DataFrame(report_data)
    if not report_df.empty:
        return report_df.sort_values(by='share_%', ascending=False).reset_index(drop=True)
    
    return pd.DataFrame(columns=['feature', 'dominant_value', 'share_%', 'unique_count'])

def get_stacking_weights(stacking_model):
    """Извлекает веса базовых моделей из мета-регрессора (Ridge/Lasso)."""
   
    meta_model = stacking_model.final_estimator_
    base_models = [name for name, _ in stacking_model.estimators]
    
    weights = pd.DataFrame({
        'Model': base_models,
        'Weight': meta_model.coef_
    }).sort_values(by='Weight', ascending=False)
    
    return weights

def run_search(estimator, param_grid, cv, scoring, X_train, y_train, n_jobs=1):
    """
    Универсальный оберточный метод для запуска GridSearchCV.
    """
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit='neg_mean_squared_error', 
        verbose=1
    )
    search.fit(X_train, y_train)
    return search

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def get_residuals_analysis(y_true, y_pred):
    """Возвращает DataFrame с истинными значениями, предсказаниями и остатками."""
    residuals = y_true - y_pred
    return pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Residuals': residuals,
        'Abs_Error': np.abs(residuals)
    })

def get_important_categories(df, cat_cols, target='SalePrice', top_n=10):
    impact = {}
    for col in cat_cols:
        # Группируем цены по категориям
        groups = [group[target].values for name, group in df.groupby(col)]
        # Проводим тест ANOVA (проверяем, равны ли средние в группах)
        if len(groups) > 1:
            f_stat, p_val = stats.f_oneway(*groups)
            impact[col] = f_stat # Чем выше F-статистика, тем сильнее различие
            
    return pd.Series(impact).sort_values(ascending=False).head(top_n).index

def check_imputation_completeness(df: pd.DataFrame):
    """Проверка, что после всех MissingValueImputer не осталось NaN."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("⚠️ Обнаружены пропуски после импутации:")
        print(missing[missing > 0])
    else:
        print("✅ Данные полностью заполнены.")

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)

def get_mi_scores(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    """
    Считает Mutual Information для признаков (числовых и категориальных).
    """
    X = X.copy()
    
    # 1. Обработка категориальных признаков (Factorize)
    # MI требует числа, поэтому превращаем текст в коды категорий (0, 1, 2...)
    for colname in X.select_dtypes(["object", "category", "string"]):
        X[colname], _ = X[colname].factorize()
        
    # 2. Обработка пропусков
    # MI не принимает NaN. Заполняем 0 (для категорий это станет отдельным кодом)
    X = X.fillna(0)
    
    # 3. Создание маски дискретных признаков
    # Дискретными считаем те, что были объектами или имеют целочисленный тип
    discrete_features = [ptypes.is_integer_dtype(t) for t in X.dtypes]

    # 4. Расчет MI
    # Передаем маску discrete_features, чтобы расчет был точным
    mi_scores = mutual_info_regression(
        X, y, 
        discrete_features=discrete_features, 
        random_state=random_state
    )
    
    # Оформляем результат
    mi_series = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    return mi_series.sort_values(ascending=False)

def calculate_vif(df: pd.DataFrame): #посчитаем мультиколиинеарность между признаками
    numeric_df = df.select_dtypes(include=[np.number])
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns

    vif_data['VIF'] = [variance_inflation_factor(numeric_df.values, i) for i in range(len(numeric_df.columns))]
    vif_data['VIF'] = vif_data['VIF'].round(2)
    return vif_data.sort_values(by ='VIF', ascending=False)



def get_regression_metrics(y_true, y_pred, name="Model"):
    """Возвращает словарь с основными метриками регрессии."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": name,
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4)
    }
#проверка коллинеарнсоти у категориальных столбцов 
def check_categorical_redundancy(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Проверка избыточности категориальных признаков.
    Находит пары колонок, которые дублируют друг друга.
    """
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    redundant_pairs = []

    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            col1, col2 = cat_cols[i], cat_cols[j]
            
            match_ratio = (df[col1].astype(str) == df[col2].astype(str)).mean()
            
            if match_ratio > threshold:
                redundant_pairs.append({
                    'feat1': col1,
                    'feat2': col2,
                    'match_ratio': round(match_ratio, 4),
                    'uniques_f1': df[col1].nunique(),
                    'uniques_f2': df[col2].nunique()
                })
    
    return pd.DataFrame(redundant_pairs).sort_values(by='match_ratio', ascending=False)


def get_feature_importance(pipeline, feature_names):
    """
    Универсальный экстрактор важности признаков для любой модели.
    """
    model = pipeline.named_steps['model']
    
    # 1. Если это линейная модель (Lasso, Ridge, и т.д.)
    if hasattr(model, 'coef_'):
        importance_values = model.coef_
        
    # 2. Если это дерево или ансамбль (RF, HGBR, XGBoost)
    elif hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
    
    # 3. Специфика для HistGradientBoosting (у него важность считается иначе)
    else:
        # Если модель не отдает важность напрямую, возвращаем пустой DF
        print(f"Модель {type(model).__name__} не поддерживает прямой вывод важности.")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values,
        'abs_importance': np.abs(importance_values)
    })
    
    # Сортируем по убыванию влияния
    return importance_df.sort_values(by='abs_importance', ascending=False)

def print_grid_search_results(grid_search, model_name="Model"):
    idx = grid_search.best_index_
    results = grid_search.cv_results_
    
    # Достаем метрики
    mse = -results['mean_test_neg_mean_squared_error'][idx]
    rmse = np.sqrt(mse)
    r2 = results['mean_test_r2'][idx]
    mae = -results['mean_test_neg_mean_absolute_error'][idx]
    
    print(f"--- РЕЗУЛЬТАТЫ {model_name} ---")
    print(f"Честный MSE на валидации (CV):  {mse:.4f}")
    print(f"Честный RMSE на валидации (CV): {rmse:.4f}")
    print(f"Честный r2 на валидации (CV):   {r2:.4f}")
    print(f"Честный MAE на валидации (CV):  {mae:.4f}")
    print(f"Лучшие параметры: {grid_search.best_params_}")

def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    
    result: Dict[str, pd.DataFrame] = {}
 
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    for name in cat_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        
        if vc.empty:
            continue
            
        share = vc / vc.sum()
        result[name] = pd.DataFrame({
            "value": vc.index.astype(str),
            "count": vc.values,
            "share": share.values,
        })

    return result


def calculate_cv_permutation_importance(model, X, y, cv, n_repeats=5, n_jobs=2):
    """
    Рассчитывает усредненную Permutation Importance по всем фолдам.
    """
    all_importances = []
    
    for train_idx, val_idx in cv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Клонируем и обучаем на тренировочной части фолда
        model_fold = clone(model)
        model_fold.fit(X_tr, y_tr)
        
        # Считаем важность на валидационной части
        res = permutation_importance(
            model_fold, X_val, y_val, 
            n_repeats=n_repeats, random_state=42, n_jobs=n_jobs
        )
        all_importances.append(res.importances_mean)
        
    return np.mean(all_importances, axis=0), np.std(all_importances, axis=0)

def print_cv_results(cv_results, model_name="Model"):
    print(f"--- РЕЗУЛЬТАТЫ {model_name} ---")
    print(f"Честный RMSE на CV: {np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()):.4f}")
    print(f"Честный MSE на CV:  {(-cv_results['test_neg_mean_squared_error'].mean()):.4f}")
    print(f"Честный r2 на CV:   {cv_results['test_r2'].mean():.4f}")
    print(f"Честный MAE на CV:  {-cv_results['test_neg_mean_absolute_error'].mean():.4f}")

def compute_quality_flags(summary, missing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Эвристики «качества» данных.
    """
    flags: Dict[str, Any] = {}
    flags["too_few_rows"] = summary.n_rows < 100

    # 1. --- Пропуски ---
    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    avg_missing_share = sum(c.missing_share for c in summary.columns) / summary.n_cols if summary.n_cols > 0 else 0.0
    
    flags["max_missing_share"] = max_missing_share
    flags["avg_missing_share"] = avg_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5
    
    # имена колонок, где пропусков > 90%
    flags["cols_to_drop_missing"] = [c.name for c in summary.columns if c.missing_share > 0.9]

    # 2. --- Константные признаки ---
    # список, чтобы пайплайн мог их дропнуть
    constant_cols = [c.name for c in summary.columns if c.unique == 1]
    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["cols_to_drop_constant"] = constant_cols 

    # 3. --- Дубликаты ID ---
    id_col = next((c for c in summary.columns if c.name.lower() in ['id', 'user_id']), None)
    flags["has_suspicious_id_duplicates"] = (id_col.unique < summary.n_rows) if id_col else False

    # 4. --- Скор качества ---
    score = 1.0
    score -= avg_missing_share  # Чем больше пропусков, тем хуже
    
    if flags["too_few_rows"]:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.1 # Штрафуем за мусорные константные колонки

    flags["quality_score"] = max(0.0, min(1.0, score))

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: list[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)

logger = logging.getLogger(__name__)

def init_project_structure():
    """Создает необходимые директории и логирует процесс."""
    for folder in FOLDERS_TO_CREATE:
        if folder.exists():
            logger.info(f"Директория уже существует: {folder}")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана директория: {folder}")