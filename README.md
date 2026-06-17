# DukeOfTokyo
tweedejaarsproject KI

## Setup
Virtual environment:
1. uv venv
2. source .venv/bin/activate
3. uv pip install -r pyproject.toml

Data:
1. Create a folder 'data/'
2. Place the relevant csv files in this folder
3. Edit the config you will use to refer to the correct file (data, data_path)


## Training a single model
1. Define a model config using the yaml file as outlined there
2. Activate the venv
3. Run "python src/train.py --config [path to your config]"
4. If you want to train more models in sequence, append more filepaths to the configs you want to train, a single space seperating each path.
5. If you want to examine training progress in more detail, make sure tensorboard is installed. Then run "tensorboard --logdir experiments/tb_logs"

## Using trained model for inference
1. activate venv
2. run the following: python src/darts_prophet/inference.py --config [relative filepath to your config] --horizon [how many full hours ahead should be predicted, default 336] --offset [how many hours after the end of the training data should the first inference be made, default 0]

## Testing a single model
1. Define a model config using the yaml file as outlined there
2. Activate the venv
3. Run "python src/single_model_test.py --config [path to your config]"
4. Wait for training to be complete and graphs be generated. Examine the terminal output for further information and additional metrics.