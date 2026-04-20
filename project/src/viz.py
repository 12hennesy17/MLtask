import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor

def plot_target_distribution(y: pd.Series, title: str = "Distribution SalePrice",  save_path: str = None):
    """Рисует гистограмму и KDE для целевой переменной."""
    skewness = y.skew()
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, color='teal', label=f'Skewness: {skewness:.2f}')
    plt.title(title)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_missing_values(df: pd.DataFrame, save_path: str = None):
    """Визуализация пропусков в данных (только тех, где они есть)."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if missing.empty:
        print("✅ Пропусков нет!")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing.values, y=missing.index, palette='flare', hue=missing.index)
    plt.title("Пропущенные значения по признакам")
    plt.xlabel("Количество NaN")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str = "SalePrice", save_path: str = None):
    """Рисует корреляцию всех числовых признаков с таргетом."""
    corr = df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
    
    plt.figure(figsize=(8, 12))
    sns.heatmap(corr.to_frame(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation features with {target_col}")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()



def plot_mi_scores(
    mi_scores: pd.Series, 
    features: list = None, 
    threshold: float = 0.4, 
    mode: str = 'strong', 
    save_path: str = None
):
    """
    Визуализация MI Scores.
    - features: список конкретных признаков для отрисовки (если None, работает фильтр по threshold)
    - mode='weak': показывает признаки ниже или равные threshold
    - mode='strong': показывает признаки строго выше threshold
    """
    
    # 1. Если передали конкретный список признаков
    if features is not None:
        # Берем только те признаки из списка, которые есть в mi_scores
        scores = mi_scores.loc[mi_scores.index.isin(features)]
        title = f"MI Scores for selected features"
        palette = 'viridis'
    
    # 2. Иначе фильтруем по порогу (старая логика)
    elif mode == 'weak':
        scores = mi_scores[mi_scores <= threshold]
        title = f"Weak features (MI <= {threshold})"
        palette = 'magma'
    else:
        scores = mi_scores[mi_scores > threshold]
        title = f"Strong features (MI > {threshold})"
        palette = 'rocket'

    if scores.empty:
        print(f"Признаков для отображения не найдено.")
        return

    # Сортируем для красоты графика
    scores = scores.sort_values(ascending=False)

    plt.figure(figsize=(10, max(4, len(scores) * 0.4)))
    sns.barplot(x=scores.values, y=scores.index, hue=scores.index, palette=palette, legend=False)
    
    plt.title(title)
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_missing_correlation(df: pd.DataFrame, save_path: str = None):
    """Тепловая карта корреляции отсутствия данных."""
    ax = msno.heatmap(df)
    ax.set_title("Корреляция пропущенных значений", fontsize=16)
    
    if save_path:
        ax.figure.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_categorical_impact(df, categories, target='SalePrice', figsize=(16, 12), save_path: str = None):
    """
    Рисует боксплоты для списка категориальных признаков относительно таргета.
    """
    n_rows = (len(categories) + 3) // 4  
    plt.figure(figsize=(16, 4 * n_rows))
    
    for i, col in enumerate(categories):
        plt.subplot(n_rows, 4, i + 1)     
        
        sns.boxplot(x=col, y=target, data=df, hue=col, legend=False)
        plt.axhline(df[target].median(), color='red', linestyle='--', alpha=0.6)
        
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Impact of {col} on {target}', fontsize=14)
        plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_top_scatter(df: pd.DataFrame, target_col: str, top_n: int = 6, save_path: str = None):
    """Матрица графиков рассеяния для признаков с самой высокой корреляцией."""
    
    corr = df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
    top_features = corr.index[1:top_n+1] 
    
    rows = (top_n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(top_features):
        sns.scatterplot(data=df, x=col, y=target_col, ax=axes[i], alpha=0.6)
        axes[i].set_title(f'{target_col} vs {col}', fontsize=12)
        axes[i].grid(True, alpha=0.3)

    # Удаляем пустые оси
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_rf_convergence(best_params, X, y, range_n=(50, 500, 20), save_path=None):
    # Копируем параметры, чтобы не испортить оригинал
    params = best_params.copy()
    if 'n_estimators' in params:
        params.pop('n_estimators')
        
    n_estimators_range = range(*range_n)
    oob_scores = []

    for n in n_estimators_range:
        model = RandomForestRegressor(
            n_estimators=n, **params, oob_score=True, random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        oob_scores.append(model.oob_score_)

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, oob_scores, marker='o', linestyle='-', color='b')
    plt.title('Convergence: OOB Score vs Number of Trees')
    plt.xlabel('Number of trees')
    plt.ylabel('OOB Score (R^2)')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_lasso_cv_results(cv_results, alphas, best_alpha, title="Lasso Validation Curve"):
    train_rmse = np.sqrt(-cv_results['mean_train_neg_mean_squared_error'])
    test_rmse = np.sqrt(-cv_results['mean_test_neg_mean_squared_error'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, train_rmse, label='Train RMSE', linestyle='--')
    plt.plot(alphas, test_rmse, label='Validation RMSE (CV)', linewidth=2)
    plt.axvline(best_alpha, color='green', linestyle=':', label=f'Best Alpha: {best_alpha:.4f}')
    
    plt.xscale('log')
    plt.xlabel('Alpha (Log scale)')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15, title="Feature Importance", save_path=None):
    # Универсально: берем либо коэффициенты, либо важность
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print("Модель не поддерживает вывод важности признаков!")
        return

    # Создаем DataFrame для удобства
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=True).tail(top_n)

    # Рисуем
    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp['feature'], feat_imp['importance'], color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_lasso_coefficients(model, feature_names, top_n=15, title="Lasso Top Features", save_path=None):
    """
    Визуализирует самые важные коэффициенты Lasso-регрессии.
    """
    # 1. Извлекаем коэффициенты
    lasso_coefs = model.coef_

    # 2. Собираем всё в удобную таблицу
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lasso_coefs,
        'Abs_Coefficient': np.abs(lasso_coefs)
    })

    # Оставляем только те признаки, которые Lasso не занулила
    non_zero_coefs = coef_df[coef_df['Coefficient'] != 0]
    
    print(f"Lasso занулила {len(feature_names) - len(non_zero_coefs)} признаков из {len(feature_names)}.")

    # Берем топ-N самых "тяжелых" признаков
    top_features = non_zero_coefs.sort_values(by='Abs_Coefficient', ascending=False).head(top_n)

    # 3. Строим график
    plt.figure(figsize=(10, top_n * 0.5 + 1)) # Динамическая высота
    colors = ['seagreen' if c > 0 else 'lightcoral' for c in top_features['Coefficient']]

    # Рисуем горизонтальный Bar plot
    plt.barh(y=top_features['Feature'][::-1], width=top_features['Coefficient'][::-1], color=colors[::-1], edgecolor='black', alpha=0.8)

    plt.title(title, fontsize=14)
    plt.xlabel('Influence on Price (Coefficient)', fontsize=12)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Добавляем подписи значений
    for index, value in enumerate(top_features['Coefficient'][::-1]):
        plt.text(value, index, f' {value:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"График сохранен по пути: {save_path}")
        
def plot_hgbr_learning_curve(model, grid_search_params, save_path=None):
    """
    Визуализирует кривые обучения для HistGradientBoostingRegressor.
    """
    train_errors = model.train_score_
    val_errors = model.validation_score_
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label='Train Loss')
    plt.plot(val_errors, label='Validation Loss')
    
    plt.title(f'HGBR Convergence\nParams: {grid_search_params}')
    plt.xlabel('Iterations (Trees)')
    plt.ylabel('Loss')
    plt.axvline(model.n_iter_ - 1, color='green', linestyle=':', label='Best Iteration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_permutation_importance(importance_df, model_name="Model", save_path=None):
    """
    График важности перестановок (Permutation Importance).
    Принимает DataFrame из core.get_cv_permutation_importance.
    """
    plt.figure(figsize=(12, 8))
    plt.barh(y=importance_df['feature'][::-1], 
             width=importance_df['importance'][::-1], 
             xerr=importance_df.get('std', 0)[::-1], 
             color='skyblue', edgecolor='navy', alpha=0.8)

    plt.title(f"Permutation Importance {model_name} (Cross-Validation)", fontsize=14)
    plt.xlabel("Снижение R² на валидационных данных")
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.show()

def plot_pdp_top_features(model, X, feature_names, importance_df, top_n=5, save_path=None):
    """Автоматический PDP для топ-N признаков из таблицы важности."""
    top_idx = importance_df.head(top_n).index
    # Находим индексы признаков в исходной матрице X
    # (Если X - numpy, индексы берем из importance_df напрямую)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    PartialDependenceDisplay.from_estimator(
        model, X, features=top_idx, 
        feature_names=feature_names, ax=ax, grid_resolution=50
    )
    plt.suptitle(f"Partial Dependence для топ-{top_n} факторов", fontsize=16)
    
    if save_path:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
    plt.show()

def plot_stacking_weights(meta_weights, model_names, save_path=None):
    """Вклад базовых моделей в Stacking."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=meta_weights, hue=model_names, palette='viridis', alpha=0.8)
    plt.title('Вклад моделей в финальный предсказание (Stacking Weights)', fontsize=13)
    plt.ylabel('Вес (Коэффициент мета-регрессора)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, r2_score, save_path=None):
    """Финальный Sanity Check: сравнение факта и прогноза."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='purple')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)

    plt.xlabel('Реальная логарифмическая цена')
    plt.ylabel('Предсказанная цена')
    plt.title(f'Качество модели: R² = {r2_score:.4f}', fontsize=14)
    plt.grid(True, alpha=0.2)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_train_test_drift(df_train, df_test, features, save_path=None):
    """
    Сравнивает распределение признаков в обучающей и тестовой выборках.
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.kdeplot(df_train[col], label='Train', fill=True, alpha=0.4, ax=axes[i], color='#1f77b4')
        sns.kdeplot(df_test[col], label='Test', fill=True, alpha=0.4, ax=axes[i], color='#ff7f0e')
        axes[i].set_title(f'Distribution Comparison: {col}', fontsize=14)
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3)

    # Удаляем пустые сабплоты, если количество признаков нечетное
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    
    plt.show()