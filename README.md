# Скальперская система

Проект реализует архитектуру скальпирующей торговой платформы в соответствии с ТЗ.

## Основные компоненты

- `scalp_system.config` — описание конфигурации и загрузчик YAML/JSON.
- `scalp_system.data` — потоковая подписка и DataEngine с TTL-кешем, кольцевыми буферами и ротацией инструментов.
- `scalp_system.features` — генерация фич из стакана.
- `scalp_system.ml` — ансамбль моделей (LSTM, GBDT, Transformer, SVM) и формирование сигналов.
- `scalp_system.ml.training` — оффлайн-тренировка ансамбля и подготовка весов моделей.
- `scalp_system.risk` — риск-менеджмент и контроль лимитов.
- `scalp_system.execution` — исполнение заявок через API брокера.
- `scalp_system.monitoring` — детектор дрейфа и метрики.
- `scalp_system.monitoring.audit` — аудит действий в формате W3C.
- `scalp_system.monitoring.resource` — контроль загрузки CPU/GPU/памяти.
- `scalp_system.ml.calibration` — постановка задач калибровки моделей.
- `scalp_system.storage` — SQLite репозиторий для логирования сигналов.
- `scalp_system.utils.integrity` — проверки целостности данных при переподключении.
- `scalp_system.security` — менеджер ключей шифрования токенов.

## Запуск

```bash
pip install --no-index tinkoff_investments-0.2.0b117-py3-none-any.whl  # локальная установка SDK
python -m scalp_system config.example.yaml
```

При отсутствии поддержки YAML можно использовать JSON-конфигурацию (`config.json`).

Для работы с зашифрованными токенами укажите путь к Fernet-ключу в секции `security`,
а сами значения пометьте префиксом `enc:`. Ключ можно создать командой:

```python
from scalp_system.security import KeyManager
key = KeyManager.generate()
print(key.serialise())
```

## Дополнительные утилиты

- `python -m scalp_system.cli.init_config --env production`
- `python -m scalp_system.cli.model_trainer calibrate --config config.example.yaml`
- `python -m scalp_system.cli.model_trainer train --dataset data/training.jsonl --output runtime/models`
- `python -m scalp_system.cli.health_check --config config.example.yaml`

Секция `training` конфигурации описывает путь до JSONL-датасета с историческими примерами,
параметры обучения (число эпох, скорость обучения, размер валидационной выборки) и место
сохранения артефактов. Команда `train` формирует файл `ensemble_weights.json` в указанной
директории, создает отчёт `*.training.json` с метриками и генерирует план квантования
`quantization_plan.json` (INT8 для весов выше порога, FLOAT16 для остальных моделей).

## Мониторинг и калибровка

- Метрики дрейфа сохраняются в `runtime/drift_metrics/drift_metrics_YYYYMMDD.jsonl`.
- Триггеры на калибровку пишутся в `runtime/calibration_queue.jsonl` с дедупликацией.
- Резервные копии сигналов сохраняются в `runtime/signals.fallback.jsonl`, кеш состояния сбрасывается адаптивно с интервалом 1–5 секунд в зависимости от нагрузки.
- Перезагрузка моделей очищает кэш фич, валидирует TFLite файлы и уведомляет RiskEngine.
- Аудит действий пишется в формате W3C (`runtime/audit.log`) с тегами `ORDER`, `RISK`, `RESOURCE`.
- Монитор ресурсов переводит систему в упрощённый режим при превышении лимитов CPU/GPU/памяти.
