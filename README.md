# Notes

## Model training

### DONE

| Model type    | Batch-size | Learn-rate |Shuffle | epochs | Loss    | Accuracy | AP     | is_training |
| ------------- |:----------:|:----------:|:------ |:------:|:-------:|:--------:|:------:|:----------- |
| cross-subject | 1          | 0.0005     | False  | 200    |  0.027  |  ~70 %   | ~0     | True        |
| cross-subject | 1          | 0.0005     | True   | 200    |  0.112  |    0 %   | 0      | False       |
| cross-subject | 10         | 0.0005     | False  | 200    |  0.112  |    0 %   | 0      | False       |
| cross-subject | 40         | 0.0005     | False  | 250    |  0.110  |   ~4 %   | 0      | False       |
| cross-subject | 1          | 0.001      | True   | 200    |  0.046  |  ~65 %   | 0.025  | True        |
| cross-subject | 1          | 0.01       | True   | 100    |  0.150  |  ~3  %   | ~0     | False       |
| cross-view    | 1          | 0.0005     | False  | 200    |  0.034  |  ~80 %   | 0.015  | Not-always  |

| Model type    | Batch-size | Learn-rate |Shuffle | epochs | Loss    | Accuracy | AP     | is_training |
| ------------- |:----------:|:----------:|:------ |:------:|:-------:|:--------:|:------:|:----------- |
| cross-subject | 1          | 0.005      | True   | 200    | 0.123   |    0 %   | 0      | False       |
| cross-subject | 20         | 0.001      | True   | 200    | 0.111   |    0 %   | 0      | False       |
| cross-view    | 1          | 0.001      | True   | 200    | 0.113   |    0 %   | 0      | False       |
| cross-view    | 20         | 0.001      | True   | 200    | 0.11    |    0 %   | 0      | False       |
| cross-view    | 1          | 0.0001     | True   | 200    | 0.11    |    0 %   | 0      | False       |

### TODO

| Model type    | Batch-size | Learn-rate |Shuffle | epochs | Loss    | Accuracy | AP     | is_training |
| ------------- |:----------:|:----------:|:------ |:------:|:-------:|:--------:|:------:|:----------- |
| cross-subject | 1          | 0.0001     | True   | 200    |   xx    |   xx %   | x      |             |
| cross-view    | 1          | 0.0001     | True   | 200    |   xx    |   xx %   | x      |             |
| ------------- | ---------- | ---------- | ------ | ------ | ------- | -------- | ------ | ----------- |
| cross-subject | 1          | 0.001/5 + 0.0005      | True   | 200    |         |   xx %   | x      |

Pre cross-subject scenar dotrenovat moznost batch_1_lr_001 s mensim LR, nakopirovat model nech mam aj original
