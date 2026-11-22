# Kaggle-пайплайн дообучения

Локальный сервис использует маленький MegaDescriptor-T-224 и не требует GPU.
Тяжёлая часть — сегментация частей тела (уши/ноги/тело) на DINOv3 — обучается
в Kaggle и возвращается в проект одним файлом `data/ml/horse_parts.pt`.

## 0. Что уже готово

- `tools/labeler.html` — ручная разметка полигонов (уши, ноги, тело).
- `scripts/prepare_kaggle_dataset.js` — собирает ~50 фото из каталога в `kaggle/dataset/`.
- `kaggle/notebook.ipynb` — готовый Kaggle-ноутбук: DINOv3 + линейная голова + self-training + экспорт TorchScript.
- `kaggle/train_parts_dinov3.py` — тот же пайплайн как скрипт (для CLI/GPU-серверов).

## 1. Подготовить датасет

```bash
npm run kaggle:prepare
```

Скрипт выберет ~50 датированных фото (по 3 на особь + 5 дополнительных с
максимальным разбросом дат), скопирует их в `kaggle/dataset/images/` и создаст
`labels.json` и `dataset-metadata.json`. Уже размеченный `labels.json` не
перезаписывается.

## 2. Разметить ~50 фото

1. Откройте `tools/labeler.html` в браузере.
2. «Загрузить папку с фото» → выберите `kaggle/dataset/images/`.
3. Для каждого фото обведите контуры: **уши** (красный), **ноги** (синий), **тело** (зелёный).
   Левый клик — точка, двойной клик или «Замкнуть контур» — завершить полигон.
4. «Скачать labels.json» → замените им `kaggle/dataset/labels.json`.

## 3. Загрузить в Kaggle

1. На kaggle.com: **Create → New Dataset**, название `takhi-parts`.
2. Перетащите в браузер содержимое `kaggle/dataset/` (папку `images/` и `labels.json`). Сделайте dataset **Private**.
3. Create → Done. Dataset будет виден как `YOUR_USER/takhi-parts`.

Опционально так же создайте `takhi-unlabeled` с неразмеченными кадрами для self-training.

## 4. Запустить обучение

1. **Create → New Notebook** → прикрепите dataset `takhi-parts` (и опционально `takhi-unlabeled`).
2. Accelerator → **GPU P100** (справа в настройках сессии).
3. **Add-ons → Secrets** → добавьте секрет `HF_TOKEN` (токен Hugging Face; модель DINOv3 — gated, нужно принять лицензию Meta).
4. Загрузите `kaggle/notebook.ipynb` (File → Import Notebook) или скопируйте ячейки.
5. **Run All**. В конце появится `/kaggle/working/horse_parts.pt`.

## 5. Вернуть модель в проект

Скачайте `horse_parts.pt` из `Output` ноутбука и положите:

```bash
mv ~/Downloads/horse_parts.pt data/ml/horse_parts.pt
npm start
```

В интерфейсе под загруженным снимком появится кнопка **«Части тела»**. Если
файла нет, сервис отвечает `available:false` и кнопка честно сообщает, что
модель не установлена.

## Контракт модели

Экспортированный TorchScript принимает `uint8`-тензор `[N,3,H,W]` и возвращает
логиты `[N,4,H,W]`: 0 — фон, 1 — уши, 2 — ноги, 3 — тело. Это ровно то, что
ожидает `ml/parts_segment.py`.

## Ограничения

- Псевдоразметка упрощена до прямоугольников вокруг уверенных пикселей — для
  первого цикла достаточно, ручную разметку стоит расширять.
- TorchScript-экспорт через `transformers` может упасть на отдельных версиях.
  Тогда экспортируйте ONNX или сохраните веса головы + имя backbone — контракт
  локального раннера не меняется.
- Классификация особи остаётся локальной (MegaDescriptor + каталог) и уже
  поддерживает произвольное число классов — по числу особей в каталоге.
