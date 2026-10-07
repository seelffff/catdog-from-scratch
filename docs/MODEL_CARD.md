# Model card / Паспорт модели

## Purpose / Назначение

Educational binary image classification: cat versus dog. A single-image classifier, not an object detector or animal counter. Учебная бинарная классификация фотографий кошек и собак.

## Architecture / Архитектура

Custom `robust_resnet18`, width 64, 11,286,882 trainable parameters. This is a project implementation, not the unmodified torchvision ResNet18. Three 3×3 stem convolutions, BatchNorm/ReLU, 2×2 max pooling, four stages with two residual/SE blocks each, adaptive global average pooling, dropout 0.15 and `Linear(512, 2)`.

The first convolution has shape `[32, 3, 3, 3]`: 864 weights, no bias. Layer counts are in [parameter_breakdown.json](../reports/v2/parameter_breakdown.json).

## Data and training / Данные и обучение

Only the assigned dataset: 9,972 valid images. Train 7,975; validation 997; test 1,000. Detected duplicate groups stay inside one split. Fresh random initialization; no external pretrained weights. AdamW, four warmup epochs, cosine learning-rate decay, MixUp 0.1, label smoothing 0.02, EMA 0.995 and robust augmentations. Main run: 90 epochs; selected EMA snapshot: epoch 67.

## Inference / Предсказание

EXIF orientation, RGB, bilinear letterbox to 256×256 with gray `(128,128,128)` padding. Normalize `(pixel / 255 - 0.5) / 0.5`. Class order: cat 0, dog 1. Temperature 0.8705505632961241, dog threshold 0.5. Horizontal-flip TTA disabled.

## Evaluation / Оценка

Test: 975/1,000 correct, 25 errors, accuracy 97.5%, macro F1 ≈0.9750, ROC AUC ≈0.9966. Confusion matrix `[[483,14],[11,492]]`. Validation accuracy 969/997. Approximate image-level Wilson 95% interval: 96.34%–98.30%; it does not cover distribution shift or undetected correlation between images.

All architecture, inference-resolution and calibration choices were fixed before test evaluation. See the committed experiment reports. Scores on a different earlier test split are not direct improvement estimates.

## Artifact / Файл

The repository contains [`v2_best.pt`](../models/v2_best.pt); its size and SHA-256 are in [model_manifest.json](../models/model_manifest.json). Absolute paths were removed from checkpoint metadata. Every tensor and inference setting was compared with the original evaluated artifact and remains unchanged. The published file has a different byte checksum because metadata was repacked.

## Limits / Ограничения

No reliable rejection of other species, people, objects, drawings, or empty frames. No supported interpretation for a cat and dog in the same photograph. Rare breeds and different capture conditions can cause errors. A softmax probability is a class score under this two-class model, not a certainty about an arbitrary input.
