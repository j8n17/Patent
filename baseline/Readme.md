## 파일 구조
```
data
├─  train
│   ├─  docs
│   └─  labels
├─  test
│   └─  test_input.csv
└─  sample_submission.csv

baseline
├─  config
│   └─  preprocess.yaml, train.yaml, pred.yaml
├─  modules
│   └─  loss.py
├─  preprocess.py, train.py, pred.py
├─  Readme.md
│ 
└─  results
    └─  checkpoint-#####
        └─  submission.csv
```

- preprocess.py: 전처리 코드. `train.csv`, `category.csv` 파일 생성
- train.py: 학습 코드. `./results/checkpoint-#####` 생성
- predict.py: 추론 코드. `./results/checkpoint-#####/submission.csv` 생성

- config: 전처리/학습/추론 설정 yaml 파일