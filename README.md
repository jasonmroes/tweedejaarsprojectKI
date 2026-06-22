# DukeOfTokyo
tweedejaarsproject KI

## Setup
1. uv venv, create a new virtual environment
2. source .venv/bin/activate, activate this virtual environment
3. uv pip install -r pyproject.toml, install the required modules to this environment

Data:
1. Create a folder 'data/'
2. Place the relevant csv files in this folder
3. Edit the config you will use to refer to the correct file (data, data_path)


## Testing a single model
1. Define the model using a yaml file, following the same format as the provided configs. Note 'experiment name" and "save_dir", make sure to update these to avoid them overwriting existing data or be saved in the incorrect folder.
2. During training, the paths to the past and future dataset should both point to the historical data csv. only at inference time should the future datapath point at the forecast data. It is possible to create two seperate configs, one for training and one solely for inference.
3. activate venv
4. python src/darts_experiments/single_model_test.py --config [relative path to your config]
5. Wait for training to complete. Upon completion, examine graphs and additional metrics that are printed in terminal (and saved to a json file). These are automatically saved to experiments/results/[your model name]

## Training a model
1. Before training, make sure you have tested your config once according to the steps above. If not, proceed at your own risk.

2. activate venv
3. python src/darts_prophet/train.py --config [relative path to your config]
4. If you want to train more than one config, add the relative path to the other config, seperated by a single empty space.

## Using for inference
1. If you have tested the model once as described above, and trained the 'final' version using train.py, you can perform inference.
2. activate venv
3. run the following: python src/darts_prophet/inference.py --config [relative filepath to your config] --horizon [how many full hours ahead should be predicted, default 336] --offset [how many hours after the end of the training data should the first inference be made, default 0]

4. This returns a file inference.csv . To visualise it, run visualization/visualise.py --data-csv [path to training data used] inference-csv [path to the generated inference.csv.]


