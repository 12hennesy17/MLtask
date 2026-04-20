# -*- coding: utf-8 -*-
from __future__ import annotations

# --- Стандартные библиотеки ---
import os
import json
import joblib
import logging
from pathlib import Path
from typing import Optional

# --- Библиотеки анализа и CLI ---
import pandas as pd
import numpy as np
import typer

# --- Scikit-learn: Модели ---
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import (
    StackingRegressor, 
    RandomForestRegressor, 
    HistGradientBoostingRegressor
)

# --- Локальные модули проекта ---
from .config import *
from .pipeline import get_pipeline
from .logger_setup import setup_logging

from .core import (
    # Анализ данных
    DatasetSummary,
    summarize_dataset,
    missing_table,
    flatten_summary_for_print,
    compute_quality_flags,
    correlation_matrix,
    log_transform_target,
    get_regression_metrics,
    get_mi_scores,
    clean_outliers,
    top_categories,
)

from .viz import (
    plot_target_distribution,
    plot_missing_values,
    plot_missing_correlation,
    plot_correlation_heatmap,
    plot_top_scatter,
    plot_categorical_impact,
    plot_mi_scores,
    
)

app = typer.Typer(help="Ames Housing ML Pipeline CLI")

logger = logging.getLogger('src.cli')

@app.callback()
def global_setup(log_file: str = "logs/cli.log"):
    setup_logging(log_file)
    typer.secho("Логирование инициализировано", fg='green')
    
def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        logger.error(f"Файл не найден: {path}") # <--- ЛОГ
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding)
        logger.info(f"Загружен датасет {path}: {df.shape[0]} строк, {df.shape[1]} колонок.")
        return df
    except Exception as exc: 
        logger.error(f"Ошибка чтения CSV {path}: {exc}", exc_info=True)
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc
    

@app.command()
def overview(path: Path = typer.Argument(..., help='путь к файлу')):
    """Быстрый взгляд на данные."""
    df = _load_csv(path)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))
    
@app.command()
def train(
    data_path: Path = typer.Argument(..., help="Путь к train.csv"),
    config_path: Path = typer.Option(MODELING_DIR / "best_meta_model.json", help="Путь к конфигу параметров"),
    outlier_ids: str = typer.Option("1299,524", help="ID выбросов"),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
):
    """Обучение модели на train"""
    logger.info(f"🚀 Запуск обучения. Данные: {data_path}, конфиг: {config_path}")
    # 1. Загрузка данных
    df = _load_csv(data_path, sep=sep, encoding=encoding)
    ids_to_remove = [int(i) for i in outlier_ids.split(",")]

    df = clean_outliers(df, ids_to_remove)
    X = df.drop(columns=[TARGET, 'Id'], errors='ignore')
    y = log_transform_target(df[TARGET])

  
    with open(config_path, "r") as f:
        config = json.load(f)
    
    if 'base_models' not in config:
        logger.critical("В JSON не найден ключ 'base_models'. Остановка.") # <--- ЛОГ
        typer.secho("❌ Ошибка: В JSON не найден ключ 'base_models'", fg="red")
        raise typer.Exit()

    lasso = Lasso(**config['base_models']['lasso'])
    rf = RandomForestRegressor(**config['base_models']['rf'], random_state=42)
    hgb = HistGradientBoostingRegressor(**config['base_models']['hgb'], random_state=42)


    stacking_reg = StackingRegressor(
        estimators=[('lasso', lasso), ('rf', rf), ('hgb', hgb)],
        final_estimator=Ridge(alpha=1.0) 
    )

    # 5. Обучение   
    final_pipeline = get_pipeline(model=stacking_reg)   
    logger.info("🛠 Запуск процесса обучения пайплайна...")
    final_pipeline.fit(X, y)

    preds_train = final_pipeline.predict(X)
    train_metrics = get_regression_metrics(y, preds_train)
    
    typer.secho("📊 Метрики на обучающей выборке:", fg="blue")
    for k, v in train_metrics.items():
        display_value = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        typer.echo(f"Метрика {k}: {display_value}")

    # Сохранение
    joblib.dump(final_pipeline, PRODUCTION_MODELS_DIR / 'final_production_model.joblib')
    logger.info(f"✅ Модель сохранена: {PRODUCTION_MODELS_DIR / 'final_production_model.joblib'}")


@app.command()
def predict(
    data_path: Path = typer.Argument(..., help="Путь к test.csv"),
    model_path: Path = typer.Option(PRODUCTION_MODELS_DIR / 'final_production_model.joblib', help="Путь к модели"),
    output_path: Path = typer.Option(SUBMISSIONS_DIR / "submission.csv", help="Путь для сохранения результата"),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
):
    """Генерация предсказаний для новых данных."""
    # Загрузка модели
    if not model_path.exists():
        logger.error(f"Модель не найдена по пути: {model_path}")
        typer.secho(f"❌ Модель не найдена: {model_path}", fg="red")
        raise typer.Exit()
    
    model = joblib.load(model_path)
    logger.info("Модель успешно загружена для инференса.")

    df =_load_csv(data_path, sep=sep, encoding=encoding)
    X_test = df.drop(columns=[TARGET, 'Id'], errors='ignore')
    
    preds = model.predict(X_test)
    logger.info(f"Сгенерировано {len(preds)} предсказаний.")

    output_df = pd.DataFrame({'Id': df['Id'], 'SalePrice': np.expm1(preds)}) # Если таргет логарифмировался
    output_df.to_csv(output_path, index=False)
    
    typer.secho(f"✅ Предсказания сохранены в {output_path}", fg="green")
    logger.info(f"✅ Предсказания сохранены в {output_path}")
    
@app.command()
def healthcheck(model_path: Path = PRODUCTION_MODELS_DIR / 'final_production_model.joblib',
                data_path: Path = typer.Option(TEST_DATA, help='Путь к train.csv'),):
    """Проверка целостности модели и её совместимости."""
    logger.info("Запуск проверки здоровья (healthcheck)...")
    
    if not model_path.exists():
        typer.secho(f"❌ Модель не найдена: {model_path}", fg="red")
        raise typer.Exit(code=1)
        
    try:
        model = joblib.load(model_path)
        
        # Берем 1 строку из данных, чтобы сохранить все типы (строки, числа)
        df_sample = _load_csv(data_path).drop(columns=[TARGET, 'Id'], errors='ignore')
        sample = df_sample.iloc[[0]] # Берем первую строку 
        
        # Проверяем, что предсказание работает
        _ = model.predict(sample)
        logger.info("Healthcheck пройден успешно.")
        typer.secho("✅ Модель здорова и успешно обработала данные!", fg="green")
        
    except Exception as e:
        logger.critical(f"Healthcheck провален! Причина: {e}", exc_info=True)
        typer.secho(f"❌ Модель неисправна или несовместима: {e}", fg="red")
        raise typer.Exit()
    
@app.command()
def report(
    path: Path = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: Path = typer.Option(CLI_REPORTS_DIR, help="Каталог для отчёта."),
):
    """Генерация комплексного EDA-отчёта с Markdown-документацией."""
    logger.info(f"📊 Начало генерации отчёта для {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_csv(path)

    # 1. Аналитика (расчеты)
    logger.info("⚙️ Расчет статистик...")
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    quality = compute_quality_flags(summary, missing_df)
    top_cats_dict = top_categories(df)
    mi_scores = get_mi_scores(df.drop(columns=[TARGET, 'Id'], errors='ignore'), df[TARGET])
    # Сохранение таблиц
    cat_dir = out_dir / "top_categories"
    cat_dir.mkdir(exist_ok=True)
    for col_name, df_top in top_cats_dict.items():
        df_top.to_csv(cat_dir / f"{col_name}.csv", index=False)
    
    flatten_summary_for_print(summary).to_csv(out_dir / "summary.csv", index=False)
    missing_df.to_csv(out_dir / "missing.csv")
    if not corr_df.empty:
        corr_df.to_csv(out_dir / "correlation.csv")

    # 2. Визуализация и сохранение файлов
    logger.info("🎨 Генерация графиков...")
    plot_target_distribution(df[TARGET], save_path=out_dir / "target_dist.png")
    plot_missing_values(df, save_path=out_dir / "missing_values.png")
    plot_missing_correlation(df, save_path=out_dir / "missing_corr.png")
    plot_correlation_heatmap(df, target_col=TARGET, save_path=out_dir / "corr_heatmap.png")
    plot_top_scatter(df, target_col=TARGET, save_path=out_dir / "top_scatter.png")
    
    plot_mi_scores(mi_scores, save_path=out_dir / "mi_scores.png")
    
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    if cat_cols:
        plot_categorical_impact(df, cat_cols[:8], target=TARGET, save_path=out_dir / "cat_impact.png")

    # 3. Генерация Markdown-отчёта
    md_path = out_dir / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# EDA-отчёт: {Path(path).name}\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")
        
        f.write("## Качество данных\n\n")
        f.write(f"- Оценка качества: **{quality.get('quality_score', 0):.2f}**\n")
        f.write(f"- Макс. доля пропусков в столбце: **{quality.get('max_missing_share', 0):.2%}**\n\n")
        f.write(f"- Слишком мало строк (<100): **{'Да' if quality.get('too_few_rows') else 'Нет'}**\n")
        f.write(f"- Есть дубликаты ID: **{'Да' if quality.get('has_suspicious_id_duplicates') else 'Нет'}**\n\n")
        f.write("## Визуализации\n\n")
        f.write("![Целевая переменная](target_dist.png)\n\n")
        f.write("![Пропущенные значения](missing_values.png)\n\n")
        f.write("![Корреляция](corr_heatmap.png)\n\n")
        f.write("![Важность признаков](mi_scores.png)\n\n")
        f.write("\n## Обзор признаков\n\n")
        f.write("| Колонка | Тип | Пропуски (%) | Уникальных |\n")
        f.write("|---|---|---|---|\n")
        for col in summary.columns:
            f.write(f"| {col.name} | {col.dtype} | {col.missing_share:.1%} | {col.unique} |\n")
    logger.info(f"✅ Отчёт успешно записан в {out_dir}")
    typer.secho(f"✅ Отчёт успешно записан в {out_dir}", fg='green')
if __name__ == "__main__":
    app()