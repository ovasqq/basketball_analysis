# KaliCalib (вендорная копия)

- Источник: https://github.com/CEA-LIST/KaliCalib, коммит `3b3f52b` (2022-09-19)
- Скопирован **без изменений** только `kalicalib/model_resnet.py` (архитектура сети).
  Остальной код репозитория (обучение, датасет DeepSport, deepsport-utilities) не нужен для инференса.
- Лицензия: CeCILL 2.1 (`LICENSE.txt`), совместима с GPL. При распространении
  кода, включающего этот файл, действуют её условия.
- Веса: `models/model_challenge.pth` из того же репозитория -> `weights/kalicalib_challenge.pth`
  (в git не хранятся, см. README).
- Статья: Maglo et al., "KaliCalib: A Framework for Basketball Court Registration", arXiv:2209.07795
