# Utils

Эта папка содержит вспомогательные скрипты и утилиты для проекта rmvpe-aisingers.

## Файлы

### tensorboard_logger.py

Улучшенная система логирования для тренировки RVC моделей с поддержкой:

- **Отслеживание рекордов из TensorBoard** - автоматически находит лучшую эпоху
- **Предупреждение о перетренировке** - предупреждает после 20 эпох без улучшения  
- **Точность ~99.98%** - считывает метрики напрямую из event файлов
- **Работает на любом этапе** - не требует перезапуска тренировки

#### Использование:

```python
from utils.tensorboard_logger import TensorBoardMetricsTracker, format_training_log

# Инициализация трекера
tracker = TensorBoardMetricsTracker(log_dir="/kaggle/working/rmvpe-ai/logs/mi-test")

# Получить лучшую эпоху
best_epoch, best_loss = tracker.find_best_epoch('loss/g/mel')

# Проверить перетренировку
is_overtraining = tracker.check_overtraining(current_epoch, current_loss, patience=20)

# Форматировать лог
log_message = format_training_log(
    epoch=current_epoch,
    total_epochs=total_epochs,
    mel_loss=mel_loss,
    total_loss=total_loss,
    best_mel=best_loss,
    best_epoch=best_epoch,
    is_overtraining=is_overtraining
)
print(log_message)
```

#### Пример вывода:

```
[0350/1000] Model » Эпоха 0350 (Шаг 350) || Mel: 60.25% ▸ Рекорд: 60.25% (Эпоха 350)
[0351/1000] Model » Эпоха 0351 (Шаг 351) || Mel: 60.28% ▸ Рекорд: 60.25% (Эпоха 350) [Возможна перетренировка]
```

## Установка зависимостей

```bash
pip install tensorflow pandas
```
