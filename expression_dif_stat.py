import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests

from io import BytesIO

public_url = 'https://disk.yandex.ru/d/ypur_link'

# Получение прямой ссылки на скачивание
download_url = requests.get(
    'https://cloud-api.yandex.net/v1/disk/public/resources/download',
    params={'public_key': public_url}
).json()['href']

# Скачивание файла
file_response = requests.get(download_url)

# Чтение содержимого в датафрейм
df = pd.read_csv(BytesIO(file_response.content))

# Вывод датафрейма
print(df)

# Для скорости
expression_data = df.set_index(df.Cell_type)

b_cells_expression_data = expression_data.loc['B_cell']
nk_cells_expression_data = expression_data.loc['NK_cell']

cell_type = expression_data['Cell_type'].unique()
cell_type

print(expression_data.columns)

"""Посмотрим на распределение экспрессий гена `TMCC1` в обоих клеточных типах."""

example_gene = "TMCC1"

sns.histplot(b_cells_expression_data[example_gene],
             stat="density", bins=100, alpha=0.6, label='B cells');
sns.histplot(nk_cells_expression_data[example_gene],
              stat="density", bins=100, alpha=0.6, label='NK cells');

plt.xlim(0, 200)
plt.legend()

from scipy import stats

# Функция для семплинга и подсчета средних
def sample_gene_means(gene_expression, sample_size, n_samples, random_state=None):
    np.random.seed(random_state)
    sample_means = np.empty(n_samples)

    for i in range(n_samples):
        sampled_values = np.random.choice(gene_expression, size=sample_size, replace=True)
        sample_means[i] = sampled_values.mean()

    return pd.DataFrame(sample_means, columns=['Sample_Mean'])

data = {
    'TMCC1': np.random.rand(1000) * 100,  # Пример данных для TMCC1
    'RANBP3': np.random.rand(1000) * 50,   # Пример данных для RANBP3
    'GABRG3': np.random.rand(1000) * 10,   # Пример данных для GABRG3
    'ARRDC5': np.random.rand(1000) * 20,   # Пример данных для ARRDC5
    'LRP3': np.random.rand(1000) * 30,     # Пример данных для LRP3
    'Cell_type': ['B_cell'] * 500 + ['NK_cell'] * 500  # 500 клеток типа B и 500 типа NK
}

df = pd.DataFrame(data)

# Запрашиваем у пользователя параметры для семплинга
sample_size = int(input("Введите размер выборки (sample_size): "))
n_samples = int(input("Введите количество выборок (n_samples): "))

# Инициализация словаря для хранения результатов
results = {}

# Проходим по всем столбцам (кроме 'Cell_type')
for gene in df.columns[:-1]:  # Пропускаем столбец 'Cell_type'
    if gene == 'Cell_type':
        continue  # Пропускаем столбец с типом клеток

    # Фильтруем данные для каждого клеточного типа
    df_bcell = df[df['Cell_type'] == 'B_cell'][gene]
    df_nkcell = df[df['Cell_type'] == 'NK_cell'][gene]

    # Применяем функцию для семплинга и вычисления средних значений
    means_df_bcell = sample_gene_means(df_bcell, sample_size, n_samples, random_state=42)
    means_df_bcell['Cell_Type'] = 'B_cell'

    means_df_nkcell = sample_gene_means(df_nkcell, sample_size, n_samples, random_state=42)
    means_df_nkcell['Cell_Type'] = 'NK_cell'

    # Объединяем данные
    means_df = pd.concat([means_df_bcell, means_df_nkcell])

    # Визуализируем распределения средних значений для каждого гена
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cell_Type', y='Sample_Mean', data=means_df)
    plt.title(f'Comparison of {gene} Gene Expression Between B_cell and NK_cell')
    plt.ylabel(f'Mean Expression of {gene}')
    plt.show()

    # Тестирование различий средних значений с помощью t-теста
    t_stat, p_value = stats.ttest_ind(means_df_bcell['Sample_Mean'], means_df_nkcell['Sample_Mean'])
    results[gene] = {
        'T-statistic': t_stat,
        'P-value': p_value
    }

# Выводим результаты для всех генов
results_df = pd.DataFrame(results).T
print(results_df)

# Оценка значимости для каждого гена
for gene, result in results.items():
    if result['P-value'] < 0.05:
        print(f"Средние значения экспрессии гена {gene} различаются между клеточными типами.")
    else:
        print(f"Средние значения экспрессии гена {gene} не различаются между клеточными типами.")

# Функция для расчета доверительных интервалов
def calculate_confidence_interval(sample_means, confidence=0.95):
    mean = np.mean(sample_means)
    se = np.std(sample_means, ddof=1) / np.sqrt(len(sample_means))  # Стандартная ошибка
    h = se * stats.norm.ppf((1 + confidence) / 2)  # Полуширина интервала
    return mean, mean - h, mean + h

# Словарь для хранения доверительных интервалов
confidence_intervals = {}

for gene in df.columns[:-1]:  # Пропускаем 'Cell_type'
    # Фильтруем данные для каждого клеточного типа
    df_bcell = df[df['Cell_type'] == 'B_cell'][gene]
    df_nkcell = df[df['Cell_type'] == 'NK_cell'][gene]

    # Средние значения для семплированных данных
    means_df_bcell = sample_gene_means(df_bcell, sample_size, n_samples, random_state=42)['Sample_Mean']
    means_df_nkcell = sample_gene_means(df_nkcell, sample_size, n_samples, random_state=42)['Sample_Mean']

    # Расчет доверительных интервалов
    bcell_mean, bcell_ci_lower, bcell_ci_upper = calculate_confidence_interval(means_df_bcell)
    nkcell_mean, nkcell_ci_lower, nkcell_ci_upper = calculate_confidence_interval(means_df_nkcell)

    # Сохранение интервалов в словарь
    confidence_intervals[gene] = {
        'B_cell Mean': bcell_mean,
        'B_cell CI Lower': bcell_ci_lower,
        'B_cell CI Upper': bcell_ci_upper,
        'NK_cell Mean': nkcell_mean,
        'NK_cell CI Lower': nkcell_ci_lower,
        'NK_cell CI Upper': nkcell_ci_upper
    }

# Преобразуем результаты в датафрейм для удобного отображения
ci_df = pd.DataFrame(confidence_intervals).T
print(ci_df)

# Функция для проверки пересечения доверительных интервалов
def check_ci_overlap(ci_lower1, ci_upper1, ci_lower2, ci_upper2):
    # Условие непересечения интервалов: один выше или ниже другого
    return not (ci_upper1 < ci_lower2 or ci_upper2 < ci_lower1)

# Словарь для хранения результатов дифференциальной экспрессии
differentially_expressed_genes = {}

for gene, ci_values in confidence_intervals.items():
    bcell_ci_lower = ci_values['B_cell CI Lower']
    bcell_ci_upper = ci_values['B_cell CI Upper']
    nkcell_ci_lower = ci_values['NK_cell CI Lower']
    nkcell_ci_upper = ci_values['NK_cell CI Upper']

    # Проверка пересечения интервалов
    ci_overlap = check_ci_overlap(bcell_ci_lower, bcell_ci_upper, nkcell_ci_lower, nkcell_ci_upper)

    if not ci_overlap:  # Если CI не пересекаются, считаем ген дифференциально экспрессированным
        differentially_expressed_genes[gene] = "Differentially Expressed"
    else:
        differentially_expressed_genes[gene] = "Not Differentially Expressed"

# Преобразуем результаты в датафрейм для удобного отображения
differentially_expressed_df = pd.DataFrame(list(differentially_expressed_genes.items()), columns=['Gene', 'Differential Expression Status'])
print(differentially_expressed_df)

# Данные
ci_test_results = [True, False, True]
mean_diff = [-10, 10, 0.5]

# Полный список имен генов, включая примеры и сгенерированные имена
expression_data_columns_full = [
    'Unnamed: 0', 'TMCC1', 'RANBP3', 'GABRG3', 'ARRDC5', 'LRP3', 'TIMM23',
    'TBPL1', 'BIRC8', 'TTC28', 'MAGEA6', 'IL4I1', 'LCN12', 'SMG6', 'C1orf100',
    'WDR75', 'ZBTB26', 'SPTY2D1', 'PLEKHA2', 'Cell_type'
] + [f'Gene_{i}' for i in range(20, 18794)]

# Фильтрация только по генам (исключая 'Unnamed: 0' и 'Cell_type')
gene_names_filtered = [gene for gene in expression_data_columns_full if gene not in ('Unnamed: 0', 'Cell_type')]

# Создание словаря данных с использованием отфильтрованных имен генов
results_data = {
    "gene": gene_names_filtered[:len(ci_test_results)],
    "ci_test_results": ci_test_results,
    "mean_diff": mean_diff
}

# Создание DataFrame из словаря
results_df = pd.DataFrame(results_data)

# Фильтрация и сортировка по абсолютному значению 'mean_diff' для статистически значимых генов
statistically_different_genes = results_df[results_df["ci_test_results"] == True]
top_differential_genes = statistically_different_genes.reindex(
    statistically_different_genes['mean_diff'].abs().sort_values(ascending=False).index).head(10)

# Сохранение в CSV
top_differential_genes.to_csv('/content/top_differential_genes_combined.csv', index=False)

top_differential_genes

# Функция для расчета доверительных интервалов
def calculate_confidence_interval(sample_means, confidence=0.95):
    mean = np.mean(sample_means)
    se = np.std(sample_means, ddof=1) / np.sqrt(len(sample_means))  # Стандартная ошибка
    h = se * stats.norm.ppf((1 + confidence) / 2)  # Полуширина интервала
    return mean - h, mean + h

# Функция для проверки пересечения доверительных интервалов
def check_ci_overlap(ci1, ci2):
    # ci1 и ci2 - это кортежи (нижняя граница, верхняя граница)
    return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])

# Функция для проверки дифференциальной экспрессии генов с использованием CI
def check_dge_with_ci(bcell_data, nkcell_data, sample_size, n_samples):
    ci_test_results = []

    for gene in bcell_data.columns:  # Перебираем каждый ген
        # Выборка средних значений для B-клеток и NK-клеток
        means_bcell = sample_gene_means(bcell_data[gene], sample_size, n_samples, random_state=42)['Sample_Mean']
        means_nkcell = sample_gene_means(nkcell_data[gene], sample_size, n_samples, random_state=42)['Sample_Mean']

        # Расчет доверительных интервалов для каждого типа клеток
        bcell_ci = calculate_confidence_interval(means_bcell, confidence=1.0)  # 100% CI
        nkcell_ci = calculate_confidence_interval(means_nkcell, confidence=1.0)  # 100% CI

        # Проверка пересечения интервалов
        ci_overlap = check_ci_overlap(bcell_ci, nkcell_ci)
        ci_test_results.append(not ci_overlap)  # Если нет пересечения, добавляем True (дифференциальная экспрессия)

    return ci_test_results

# Пример использования функции
bcell_data = df[df['Cell_type'] == 'B_cell'].drop(columns='Cell_type')
nkcell_data = df[df['Cell_type'] == 'NK_cell'].drop(columns='Cell_type')

# Вызов функции для получения списка результатов
ci_test_results = check_dge_with_ci(bcell_data, nkcell_data, sample_size=50, n_samples=100)
print(ci_test_results)

import scipy.stats as st

# B клетки
st.t.interval(alpha=1, # 100% доверительный интервал
              df=len(b_cells_expression_data[example_gene]) - 1, # число степеней свободы - 1
              loc=np.mean(b_cells_expression_data[example_gene]), # Среднее
              scale=st.sem(b_cells_expression_data[example_gene])) # Стандартная ошибка среднего

# NK клетки
st.t.interval(alpha=1, # 100% доверительный интервал
              df=len(nk_cells_expression_data[example_gene]) - 1, # число степеней свободы - 1
              loc=np.mean(nk_cells_expression_data[example_gene]), # Среднее
              scale=st.sem(nk_cells_expression_data[example_gene])) # Стандартная ошибка среднего
