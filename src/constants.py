import os

CONFIG_FILE_NAME = "config_model.yaml"
MODEL_FINAL_FILE_NAME = "model_final.pth"
CHECKPOINT_FILE_NAME = "last_checkpoint"
METRICS_FILE_NAME = "metrics.json"

NUMBER_OF_JOINTS = int(os.getenv("JOINTS", 25))
NUMBER_OF_AXES = int(os.getenv("AXES", 3))

LABELS = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    15: 12,
    17: 13,
    19: 14,
    20: 15,
    22: 16,
    23: 17,
    25: 18,
    28: 19,
    29: 20,
    30: 21,
    31: 22,
    32: 23,
    33: 24,
    34: 25,
    35: 26,
    36: 27,
    37: 28,
    38: 29,
    39: 30,
    40: 31,
    41: 32,
    42: 33,
    43: 34,
    44: 35,
    45: 36,
    46: 37,
    47: 38,
    48: 39,
    49: 40,
    50: 41,
    51: 42
}
