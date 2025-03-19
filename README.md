# README

## Описание
Этот скрипт предназначен для автоматического раскрашивания черно-белых фотографий с использованием библиотеки DeOldify.

### Основные возможности:
- Использует предобученную модель для раскрашивания изображений.
- Работает на CPU (без необходимости в GPU).
- Обрабатывает все изображения в указанной входной директории и сохраняет результаты в выходную директорию.
- Пропускает уже обработанные файлы, избегая дублирования работы.

## Установка и настройка

### Требуемые библиотеки
Перед использованием скрипта необходимо установить все зависимости. Убедитесь, что у вас установлен Python 3.8+.

Установите необходимые библиотеки с помощью pip:
```bash
pip install torch deoldify pathlib
```

### Подготовка моделей
Скрипт использует библиотеку DeOldify, которая требует предобученные модели. Вам может понадобиться загрузить их вручную с официального репозитория проекта.

## Использование

1. Укажите путь к входной директории с черно-белыми изображениями (переменная `photo_dir`).
2. Укажите путь к выходной директории, где будут сохраняться раскрашенные изображения (переменная `output_dir`).
3. Запустите скрипт:
```bash
python re_color.py
```

## Настройки
- `render_factor=35` — параметр, влияющий на качество раскрашивания (чем выше, тем лучше качество, но больше времени обработки).
- `watermarked=False` — отключает водяной знак на выходных изображениях.
- `compare=True` — сохраняет изображение с разделением: слева черно-белое, справа раскрашенное.

## Лицензия
Этот скрипт распространяется под лицензией Apache 2.0

