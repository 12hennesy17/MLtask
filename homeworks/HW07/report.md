# HW07 – Report

> Файл: `homeworks/HW07/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Datasets

- Выбранные датасеты
- S07-hw-dataset-01.csv
- S07-hw-dataset-02.csv
- S07-hw-dataset-03.csv

### 1.1 Dataset A

- Файл: `S07-hw-dataset-01.csv`
- Размер: 12000 строк 9 стобцов
- Признаки: числовые
- Пропуски: нет
- "Подлости" датасета: Числовые признаки в разных шкалах + шумовые признаки 

### 1.2 Dataset B

- Файл: `S07-hw-dataset-02.csv`
- Размер: 8000 строк 4 стобцов
- Признаки: числовые
- Пропуски: нет
- "Подлости" датасета: Числовые признаки в разных шкалах + шумовые признаки +  разная плотность

### 1.3 Dataset C

- Файл: `S07-hw-dataset-03.csv`
- Размер: 15000 строк 5 стобцов
- Признаки: числовые
- Пропуски: нет
- "Подлости" датасета: разная плотность +  шумовые признаки

## 2. Protocol

Опишите ваш "честный" unsupervised-протокол.

- Препроцессинг: отмасштабировал входные признаки через StandardScaler
- Поиск гиперпараметров:
  - какой диапазон/сетка параметров для KMeans (k) list(range(2,31))
  `S07-hw-dataset-01.csv`
  dbscan_eps1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
  dbscan_min_samples1 = [3, 5, 10, 15]
  `S07-hw-dataset-02.csv`
  dbscan_eps2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
  dbscan_min_samples2 = [3, 5, 10, 15]
  `S07-hw-dataset-03.csv`
  dbscan_eps3 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
  dbscan_min_samples3 = [3, 5, 10, 15]

  -подбор гипераметров для Kmeans лучший результат кластеризации по метрике silhouette.
  {'k': 2} 0.522 Первый датасет
  {'k': 2} 0.307 Второй датасет
  {'k': 3} 0.316 Третий датасет 
- Метрики: silhouette / Davies-Bouldin / Calinski-Harabasz (и как считали для DBSCAN при наличии шума)
Метрики DBSCAN для датасетов:

`S07-hw-dataset-01.csv`
Первый датасет
BEST DBSCAN (by silhouette among valid)
algo: dbscan
params: {'eps': 1.0, 'min_samples': 10}
n_clusters: 4
noise_frac: 0.001  non-noise: 11992
metrics:
  silhouette: 0.384
  davies_bouldin: 1.159
  calinski_harabasz: 9460.9

`S07-hw-dataset-02.csv`
Второй датасет
metrics:
  silhouette: 0.545
  davies_bouldin: 0.472
  calinski_harabasz: 135.0

`S07-hw-dataset-03.csv`
Третий датасет
metrics:
  silhouette: 0.373
  davies_bouldin: 0.551
  calinski_harabasz: 17.2

Расчет каждой из метрик  
sil = float(silhouette_score(X, labels))
db = float(davies_bouldin_score(X, labels))
ch = float(calinski_harabasz_score(X, labels))

labels = model.fit_predict(X_feat)
metrics = safe_cluster_metrics(X_feat, labels)

Визуализация PCA
X_scaled → StandardScaler (обязательно перед PCA)
PCA.fit_transform() → получаем 2 главных направления
plot_2d_embedding() → визуализация с цветами кластеров
Анализ: видно ли разделение на PCA — если да, KMeans подходит

## 3. Models

Перечислите, какие модели сравнивали **на каждом датасете**, и какие параметры подбирали.

Минимум (для каждого датасета):
`S07-hw-dataset-03.csv`
Первый датасет
- KMeans поиск `k` диапозон kmeans_ks = list(range(2,31)) 
  RANDOM_STATE = 42
  random_state=RANDOM_STATE, n_init="auto"

  `S07-hw-dataset-02.csv`
Первый датасет
- KMeans поиск `k` диапозон kmeans_ks = list(range(2,31)) 
  RANDOM_STATE = 42
  random_state=RANDOM_STATE, n_init="auto"

  `S07-hw-dataset-03.csv`
Первый датасет
- KMeans поиск `k` диапозон kmeans_ks = list(range(2,31)) 
  RANDOM_STATE = 42
  random_state=RANDOM_STATE, n_init="auto"

- `S07-hw-dataset-01.csv`
  {'eps': 1.0, 'min_samples': 10}
  `S07-hw-dataset-02.csv`
  {'eps': 0.8, 'min_samples': 15}
  `S07-hw-dataset-03.csv`
  {'eps': 0.8, 'min_samples': 3}


## 4. Results
`S07-hw-dataset-01.csv`
BEST KMEANS (by silhouette)
algo: kmeans
params: {'k': 2}
n_clusters: 2
metrics:
  silhouette: 0.522
  davies_bouldin: 0.685
  calinski_harabasz: 11787.0
  inertia: 48425.9

BEST DBSCAN (by silhouette among valid)
algo: dbscan
params: {'eps': 1.0, 'min_samples': 10}
n_clusters: 4
noise_frac: 0.001  non-noise: 11992
metrics:
  silhouette: 0.384
  davies_bouldin: 1.159
  calinski_harabasz: 9460.9

`S07-hw-dataset-02.csv`
BEST KMEANS (by silhouette)
algo: kmeans
params: {'k': 2}
n_clusters: 2
metrics:
  silhouette: 0.307
  davies_bouldin: 1.323
  calinski_harabasz: 3573.4
  inertia: 16588.5

BEST DBSCAN (by silhouette among valid)
algo: dbscan
params: {'eps': 0.8, 'min_samples': 15}
n_clusters: 2
noise_frac: 0.038  non-noise: 7693
metrics:
  silhouette: 0.545
  davies_bouldin: 0.472
  calinski_harabasz: 135.0

`S07-hw-dataset-03.csv`
  BEST KMEANS (by silhouette)
algo: kmeans
params: {'k': 3}
n_clusters: 3
metrics:
  silhouette: 0.316
  davies_bouldin: 1.158
  calinski_harabasz: 6957.2
  inertia: 31123.5

BEST DBSCAN (by silhouette among valid)
algo: dbscan
params: {'eps': 0.8, 'min_samples': 3}
n_clusters: 2
noise_frac: 0.001  non-noise: 14978
metrics:
  silhouette: 0.373
  davies_bouldin: 0.551
  calinski_harabasz: 17.2

### 4.1 Dataset A

Для первого датасета выбран наилучший метод из исследуемых KMeans,Высокий silhouette: 0.522 > 0.5 → хорошее качество кластеризации
Davies-Bouldin: 0.685 → лучше, чем у DBSCAN (чем ближе к 0, тем лучше)
Calinski-Harabasz: 11787 → значительно выше, чем у DBSCAN → кластеры лучше разделены
Нет шума → все точки кластеризованы
Простая интерпретация → всего 2 кластера легче анализировать 

### 4.2 Dataset B

Для второго датасета выбран наилучший метод из исследуемых DBSCAN
Качество кластеризации значительно выше:
Silhouette почти вдвое лучше Silhouette 0.545
Davies-Bouldin показывает в 2.8 раза лучшую разделимость Davies-Bouldin 0.472
Calinski-Harabasz 135 у DBSCAN
Реалистичное выделение шума:
3.8% точек помечены как шум — это может быть важно
Более естественная кластеризация:
DBSCAN нашел те же 2 кластера, но лучше их разделил
Алгоритм учитывает плотность, что часто лучше для реальных данных
### 4.3 Dataset C

Для третьего датасета выбран наилучший метод из исследуемых DBSCAN
Silhouette 0.373
Davies-Bouldin 0.551
Calinski-Harabasz 17.2
2 четких кластера с отличной разделимостью
Всего 0.1% шума — практически идеальная кластеризация
KMeans насильно разделил на 4 кластера, что может быть искусственным

## 5. Analysis

### 5.1 Сравнение алгоритмов (важные наблюдения)

KMeans "ломается" при:
Несферических кластерах (линии, кольца)
Кластерах разной плотности
Наличии шума/выбросов
→ Потому что работает на минимизации расстояния до центроидов

DBSCAN/Hierarchical выигрывают при:
Произвольной форме кластеров
Разной плотности (для иерархической)
Наличии шума (DBSCAN сам его выделяет)
→ Потому что используют локальную плотность/близость
### 5.2 Устойчивость (обязательно для одного датасета)

- проверка устойчивости делали цикл меняя random_state = 42 + run * 100 + k
- Функция запускает KMeans по сетке k-значений, для каждого k — 5 раз с разными random_state. Оценивает устойчивость через средний ARI между всеми парами запусков. Возвращает результаты для медианного (по инерции) запуска.
- Вывод:  неустойчиво
Если средний ARI < 0.9–0.95, результаты сильно зависят от инициализации центроидов.
KMeans чувствителен к начальным условиям, особенно при перекрывающихся кластерах или сложной форме.
Низкий ARI → кластеризация не воспроизводима, решение случайно.

### 5.3 Интерпретация кластеров

Смотрел профили признаков по средним/медианам в каждом кластере. Если признак сильно отличается от общего среднего — он определяет "лицо" кластера. Например, кластер "дорогих товаров" с высокой ценой и низким рейтингом.
Выводы по графику silhouette vs k:
Лучшее k — где максимум silhouette (чаще k=2-3)
"Локоть" на inertia — показывает точку уменьшения отдачи от роста k
Устойчивый результат — если silhouette > 0.5 и падение inertia резкое
Переобучение — silhouette падает при больших k, кластеры становятся искусственными

## 6. Conclusion

KMeans хрупок к выбросам и форме кластеров — требует сферичности и одинаковой плотности
DBSCAN устойчив к шуму — сам выделяет выбросы, работает с произвольной формой
Silhouette + Davies-Bouldin + Calinski-Harabasz — нужен ансамбль метрик, одна врет
Масштабирование обязательно — иначе доминируют признаки с большим разбросом
Нестабильность KMeans — нужны многократные запуски, смотрим ARI между ними
Локоть на графике inertia — лишь эвристика, часто неочевиден
Интерпретация через профили признаков — средние/медианы по кластерам показывают их "лицо"
Unsupervised-эксперимент = итеративный подбор — нет единственного правильного ответа, только оценка устойчивости и осмысленности