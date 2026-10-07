# Publication checks / Проверки перед публикацией

The portable project was checked on 2026-10-07. These checks verify packaging and functionality; they are separate from the recorded classifier evaluation.

| Check | Result |
|---|---|
| ML and data pipeline tests | 168 passed |
| Backend tests | 60 passed |
| Frontend tests | 11 passed |
| TypeScript type check | Passed |
| Production frontend build | Passed |
| Docker image build | Passed |
| Real checkpoint loaded in container | `scratch-v2` |
| Container image prediction | Valid cat/dog probabilities |
| Real activation payload | `conv1`, `relu1`, `pool1` |
| Light and dark themes | Screenshots captured from the working application |
| Portable model SHA-256 | Matches `models/model_manifest.json` |
| Portable vs original learned tensors | All tensors equal; only local-path metadata changed |

The screenshots use an example from the assigned dataset. No raw dataset archive, private user photo collection, deployment receipt, access key or live-server configuration is included in the public source bundle.

The container smoke check runs on CPU. Its prediction for the sample cat differs from macOS CPU by a few millionths in probability, which is normal floating-point variation across runtimes. The predicted class agrees.

Русский: перед публикацией проверены код ML, API, интерфейс, сборка и запуск Docker с настоящими весами. Эти проверки подтверждают работоспособность упаковки. Точность 97,5% относится к отдельному зафиксированному эксперименту, описанному в `reports/v2/evaluation.json` и `docs/METHODOLOGY.md`.
