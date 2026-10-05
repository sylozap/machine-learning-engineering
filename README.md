# Инженерия машинного обучения: классификация видов бабочек

Практическая работа №3 по дисциплине «Технологии анализа данных».
Автоматический ML-пайплайн на DVC с версионированием данных, моделей и экспериментов.

**Датасет:** [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification):
75 видов бабочек, 6499 размеченных изображений 224×224.

## Пайплайн

```
download → prepare → augment → featurize → train → evaluate
```

| Стадия      | Что делает                                                                                         | Выход                       |
|-------------|----------------------------------------------------------------------------------------------------|-----------------------------|
| `download`  | скачивает датасет с Kaggle                                                                          | `data/raw`                  |
| `prepare`   | стратифицированно делит размеченные изображения на train/val/test (70/15/15)                        | `data/prepared`             |
| `augment`   | генерирует новый датасет: по 2 аугментированные копии каждого train-изображения                     | `data/augmented`            |
| `featurize` | извлекает признаки предобученной на ImageNet CNN (EfficientNet-B0, 1280 признаков)                  | `data/features`             |
| `train`     | обучает классификатор (`logreg`, `svm`, `random_forest`) на исходных или исходных + аугментированных данных | `models/model.joblib` |
| `evaluate`  | считает accuracy / F1 / precision / recall на val и test, строит матрицу ошибок                      | `metrics/eval.json`, `plots/` |

Тестовая выборка **никогда не аугментируется**, поэтому результаты с аугментацией и без неё сравнимы.

## Воспроизведение

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Kaggle API token: ~/.kaggle/access_token (или ~/.kaggle/kaggle.json)
dvc repro
```

## Эксперименты

```bash
# один эксперимент
dvc exp run -n svm-aug -S train.model=svm -S train.use_augmented=true

# история экспериментов и сравнение
dvc exp show --only-changed
dvc exp diff <exp1> <exp2>

# выбрать модель из истории: восстановить её код, параметры, модель и метрики в рабочую копию
dvc exp apply <exp>

# получить эксперименты из GitHub
dvc exp pull origin --all
```
