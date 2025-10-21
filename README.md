# Скальперская система

Проект реализует архитектуру скальпирующей торговой платформы в соответствии с ТЗ.

## Основные компоненты

- `scalp_system.config` — описание конфигурации и загрузчик YAML/JSON.
- `scalp_system.data` — потоковая подписка и DataEngine с TTL-кешем, кольцевыми буферами и ротацией инструментов.
- `scalp_system.features` — генерация фич из стакана.
- `scalp_system.ml` — ансамбль моделей (LSTM, GBDT, Transformer, SVM) и формирование сигналов.
- `scalp_system.risk` — риск-менеджмент и контроль лимитов.
- `scalp_system.execution` — исполнение заявок через API брокера.
- `scalp_system.monitoring` — детектор дрейфа и метрики.
- `scalp_system.storage` — SQLite репозиторий для логирования сигналов.

## Запуск

```bash
pip install --no-index tinkoff_investments-0.2.0b117-py3-none-any.whl  # локальная установка SDK
python -m scalp_system config.example.yaml
```

При отсутствии поддержки YAML можно использовать JSON-конфигурацию (`config.json`).

## Дополнительные утилиты

- `python init_config.py --env production`
- `python model_trainer.py calibrate --days 30`
- `python health_check.py --config config.example.yaml`
