# Evaluation methodology / Методика проверки

## Data boundaries

The assigned dataset contains 9,972 readable photos: 4,960 cats and 5,012 dogs. Labels come from filenames. No relabeling based on model predictions was used. Exact duplicate detection compares decoded, EXIF-oriented RGB pixels; conservative near-duplicate detection combines perceptual hashes, aspect ratio and thumbnail similarity. Four near-duplicate pairs were grouped. This detection can miss heavy edits, crops or mirrors.

| Split | Cats | Dogs | Total |
|---|---:|---:|---:|
| Train | 3967 | 4008 | 7975 |
| Validation | 496 | 501 | 997 |
| Test | 497 | 503 | 1000 |

The provided manifest is the experiment identity. Its SHA-256 is `334e10c5e72f18c4cf9c11e4bd64eed3feb0bdf34a7594803ca955f0c0fb992e`. Duplicate groups do not cross splits.

## Revision provenance

An initial development experiment used a different split. V2 retains its validation split and creates a new closed test from earlier training groups with seed 9001. Earlier model weights therefore cannot initialize V2: those models had already seen the new test photos. V2 starts all weights and buffers from random/default initialization. The committed V1 manifest is provenance metadata, not a second training input.

## Selection and calibration

Train batches update the network. Validation guides checkpoint selection, inference resolution, TTA choice and temperature calibration. Candidates rank primarily by balanced accuracy, with negative log-likelihood as the tie break. The selected single model uses EMA weights, epoch 67, and inference size 256. Temperature scaling was fitted on validation, with fixed dog threshold 0.5 and no flip TTA.

The chosen configuration was frozen before test was opened. Test produced 975/1,000 correct predictions. Do not use those errors to choose another configuration and then call the same test an untouched holdout.

## Interpretation

Accuracy depends on this particular sample of images. Its Wilson interval is an approximate sampling interval, not a bound for all photographs. The model has two outputs and no trained “other” class. A high class score can occur on out-of-distribution inputs.

The published portable checkpoint differs only in path metadata; tensors, normalization, temperature, class order and threshold were verified unchanged. Existing metric reports document the evaluated weights rather than a newly tuned model.

## По-русски

Train меняет веса; validation помогает выбрать настройки; test показывает итог после фиксации выбора. Дубликаты объединяются до разделения, иначе копия учебного снимка может попасть в контроль. Новый test V2 сформирован из старого train, поэтому веса ранней модели не используются: обучение V2 начинается заново. Указанные 97,5% относятся к 1000 сохранённым тестовым фотографиям. Они не подтверждают такое же качество на любом источнике снимков.
