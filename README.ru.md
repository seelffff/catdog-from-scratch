# Catdog: кошки и собаки с нуля

[English README](README.md) · [Учебник по ML](docs/LEARNING_RU.md) · [Паспорт модели](docs/MODEL_CARD.md)

Учебный проект полного цикла: подготовить фотографии, **обучить нейросеть со случайной инициализацией**, честно проверить качество и подключить её к сайту. Готовые предобученные веса не использовались.

**Результат отдельного теста: 975 правильных ответов из 1000, точность 97,5%.** Это измерение на сохранённом разбиении, а не обещание безошибочной работы на любых фотографиях.

![Сайт в светлой теме](docs/screenshots/site-light.jpg)

## Что реализовано

- Собственная CNN с residual-блоками, squeeze-and-excitation и тремя свёртками на входе; 11 286 882 обучаемых параметра.
- Аудит изображений, объединение дубликатов, сохранённое разбиение train/validation/test, обучение, выбор модели по validation и калибровка вероятностей.
- Финальные веса с проверяемой контрольной суммой SHA-256.
- FastAPI: обработка фотографий, реальные карты первых слоёв, ограничения размера загрузок и контроль одновременных запросов.
- Vite и TypeScript: светлая/тёмная тема, история фотографий, мобильная вёрстка и объяснение вычислений.
- Предсказание из терминала, обработка ZIP, тесты, локальный запуск через Docker и материалы для самостоятельного изучения ML.

## Измеренные результаты

| Показатель | Значение |
|---|---:|
| Корректных фотографий в выданном наборе | 9 972 |
| Train / validation / test | 7 975 / 997 / 1 000 |
| Обучаемых параметров | 11 286 882 |
| Завершённых эпох обучения | 90 |
| Выбранные веса | EMA, эпоха 67 |
| Размер входа при обучении / предсказании | 224 / 256 пикселей |
| Accuracy на validation | 969 / 997 = 97,19% |
| Accuracy на test | 975 / 1000 = 97,50% |
| Macro F1 на test | 0,9750 |
| ROC AUC на test | 0,9966 |
| Приближённый 95% интервал Wilson | 96,34%–98,30% |
| Размер переносимого файла весов | 43,14 MiB |

Метки: **`0 = кошка`, `1 = собака`**. Собака выбирается при `P(dog) >= 0.5`.

![Кривые обучения](docs/figures/training_curves.png)

Матрица ошибок: строки соответствуют истинному классу, столбцы — ответу сети.

| Истина / предсказание | Кошка | Собака |
|---|---:|---:|
| Кошка | 483 | 14 |
| Собака | 11 | 492 |

Исходные результаты: [evaluation.json](reports/v2/evaluation.json), [аудит данных](reports/v2/data_audit.json), [конфигурация обучения](reports/v2/training/robust224_w64/config.json), [методика проверки](docs/METHODOLOGY.md). В ранней версии использовался другой test: её процент нельзя напрямую считать результатом сравнения на одинаковых изображениях.

## Быстро открыть сайт

Нужны Python 3.11+, Node.js 22.12+ либо Docker с Compose. Повторять обучение для просмотра сайта не требуется.

```bash
git clone https://github.com/seelffff/catdog-from-scratch.git
cd catdog-from-scratch
python scripts/download_model.py
docker compose up --build
```

Открой **http://localhost:8000**. Один контейнер отдаёт сайт и API; веса подключаются только для чтения. Файл модели уже включён в репозиторий и [версию v1.0.0](https://github.com/seelffff/catdog-from-scratch/tree/v1.0.0). Скрипт проверяет существующий файл и может восстановить его из этой версии, если веса удалены.

Запуск без Docker:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[training,dev]'
python -m pip install -e 'backend[dev]'
python scripts/download_model.py
cd frontend
npm ci
VITE_API_MODE=live npm run build
cd ..
MODEL_PATH=models/v2_best.pt STATIC_DIR=frontend/dist \
  uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

В Windows активация выполняется через `.venv\Scripts\activate`, а переменные окружения задаются средствами выбранного терминала. Если `MODEL_PATH` не указан, API честно сообщает о режиме заглушки; команды выше подключают настоящую модель.

```bash
curl http://localhost:8000/api/health
curl -F 'file=@photo.jpg' http://localhost:8000/api/predict
```

## Предсказания без сайта

```bash
python -m catsdogs.predict photo.jpg --checkpoint models/v2_best.pt --device cpu
python scripts/predict_zip.py photos.zip --checkpoint models/v2_best.pt \
  --device cpu --output-dir outputs/photos
```

Пакетный скрипт читает фотографии прямо из ZIP, пропускает служебные записи macOS, проверяет уникальность числовых ID и сохраняет CSV с колонками `id,label`. ID берётся из имени: `123.jpg` → `123`. На Mac с Apple Silicon можно выбрать `--device mps`. Повреждённое изображение вызывает понятную ошибку, а не выдуманное предсказание.

## Как повторить обучение

Фотографии не включены в Git. Для воспроизведения нужен исходный выданный набор: файлы `cat.N.jpg` и `dog.N.jpg` в `data/raw/train/`. Используй сохранённое [разбиение V2](data/splits_v2.csv). `--skip-audit` у подготовки данных сохраняет этот manifest, а не создаёт новое разбиение.

```bash
python scripts/prepare_data.py --skip-audit
python -m catsdogs.train --manifest data/splits_v2.csv \
  --output artifacts/reproduction --device mps --epochs 90 \
  --image-size 224 --batch-size 48 --workers 2 --width 64 \
  --architecture robust_resnet18 --augmentation robust --dropout 0.15 \
  --label-smoothing 0.02 --lr 0.001 --min-lr 0.000005 \
  --weight-decay 0.0003 --warmup 4 --patience 25 --min-epochs 60 \
  --seed 9001 --mixup 0.1 --ema-decay 0.995 --max-minutes 180
```

Если MPS недоступен, используй `--device cpu`; выполнение будет медленнее. В завершённом эксперименте размер 256 для предсказаний выбран по validation. Для нового обучения это исходный рецепт; другие настройки сравнивай на validation.

```bash
python scripts/set_inference_size.py --checkpoint artifacts/reproduction/best.pt \
  --image-size 256 --manifest data/splits_v2.csv \
  --output artifacts/reproduction/infer256.pt
python -m catsdogs.evaluate --checkpoint artifacts/reproduction/infer256.pt \
  --manifest data/splits_v2.csv --output reports/reproduction \
  --device mps --finalize --model-output models/reproduction.pt
```

`--finalize` выбирает калибровку и TTA на validation, фиксирует решение и только затем открывает test. Для каждого нового опыта нужна отдельная выходная папка. Повторное обучение на разных версиях библиотек/GPU не обязано давать побитово одинаковые веса; точный файл для предсказаний опубликован отдельно.

## Учебный маршрут

В [подробном учебнике](docs/LEARNING_RU.md) разобраны тензоры, параметры, логиты, cross-entropy, производные и backpropagation, свёртки, residual-блоки, AdamW, EMA, аугментации, переобучение, метрики и калибровка. Есть ручные расчёты, короткий код и упражнения с разборами. [English guide](docs/LEARNING_EN.md) даёт тот же основной маршрут на английском.

![Сайт в тёмной теме](docs/screenshots/site-dark.jpg)

Первые карты Conv/ReLU/Pooling получены от настоящей сети. Последующие анимации и фиксированные демонстрационные фильтры — учебные иллюстрации; интерфейс объясняет это рядом с визуализациями.

## Структура

```text
backend/       FastAPI, адаптер модели и тесты API
frontend/      Vite, TypeScript, интерфейс и тесты
src/catsdogs/  Архитектура, обработка изображений, обучение и оценка
scripts/       Подготовка данных, скачивание модели и обработка ZIP
data/          Разбиения; исходные фотографии остаются локально
models/        Финальные обученные веса и контрольная сумма
reports/v2/    Результаты эксперимента и журнал обучения
docs/          Учебники, паспорт модели, графики и скриншоты
tests/         Тесты ML и подготовки данных
```

## Проверки

```bash
python -m pytest tests -q
cd backend && python -m pytest -q && cd ..
cd frontend && npm run typecheck && npm test && npm run build
```

Тесты проверяют группировку дубликатов, границы разбиений, слои модели, правила выбора кандидатов, обработку изображений, ошибки загрузок и поведение интерфейса. Они не запускают полное обучение.

## Ограничения и дальнейшая работа

Классификатор рассчитан на фотографию кошки либо собаки. Он не умеет надёжно отклонять машину, человека или пустой кадр и не рассчитан на снимок с обоими животными. Большой softmax-процент не гарантирует правильность ответа. Редкие породы, необычный свет и стиль изображения могут отличаться от тренировочных данных.

Следующие содержательные эксперименты: отдельно проверить эффект аугментаций, собрать независимую разметку сложных пород, изучить калибровку и проверить изображения вне задачи. Настройки выбираются по validation; новый test раскрывается после фиксации выбора.
