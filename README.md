# Diploma Thesis - Action Detection

Source code for my diploma thesis with topic - Action Detection in Human Motion Data using LSTM Networks

## Quick setup
- Clone this repository to your local machine `$ git clone https://github.com/NoName115/Diploma-thesis.git`
- Install python dependencies, at least python3.7 64-bit version is required
  - for running models on CPU - `$ pip install -r requirements-cpu.txt`
  - for running models on GPU (recommended) - `$ pip install -r requirements-gpu.txt`
- Select one of two branches `master` or `train-part`
  - `$ git checkout train-part`

## Configuration
- default configuration file for experiments is located in `src/config_model.yaml`
- for every experiment it's recommended to create a copy of this file and edit it based on experiment's parameters

## Run
- To see all required arguments for running the experiment, run:
  - ```shell
    ~$ PYTHONPATH=. python src/train.py --help
    ```
- Here is an example of running an experiment:
  - ```shell
    ~$ PYTHONPATH=. python src/train.py \
    --data-actions data/actions-single-subject-all-POS.data \
    --data-sequence data/sequences-single-subject-all-POS.data \
    --meta data/meta/cross-subject.txt --epochs 200 \
    --config experiments/cs_batch_1_lr_00005_junk_20_10.yaml \
    --output output_models/
    ```
- The whole process of training and evaluation of the model can be seen in real-time using TensorBoard tool:
  - ```shell
    ~$ tensorboard --logs "<output folder of the experiment>"
    ```
