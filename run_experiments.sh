#!/usr/bin/env bash
# Серия экспериментов: 3 модели × (исходные данные | исходные + аугментированные)
set -euo pipefail

for aug in false true; do
  suffix=$([ "$aug" = true ] && echo "aug" || echo "base")
  for model in logreg svm random_forest; do
    dvc exp run -n "${model//_/-}-${suffix}" -S train.model="$model" -S train.use_augmented="$aug"
  done
done

dvc exp show --only-changed --drop 'Created|State|Executor'
