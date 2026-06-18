# DukeOfTokyo
tweedejaarsproject KI

## Setup
1. uv venv
2. source .venv/bin/activate
3. uv pip install -r pyproject.toml

Data:
1. Create a folder 'data/'
2. Place the relevant csv files in this folder
3. Edit the config you will use to refer to the correct file (data, data_path)




## Recommended workflow
1. Validate a config with `single_model_test.py` first — its test-set metrics are the
   only honest, untouched generalization estimate in this pipeline (val is used for
   early stopping during training, so val metrics elsewhere are biased optimistic).
2. Only once a config is trusted, run `train.py` on the same config to retrain on
   more data (train+val) for the checkpoint that actually gets deployed.

Note: during training, weather covariates (temp_avg, rain_mm, sun_hours) are loaded
as realized/observed values from the historical CSV, not as forecasts. At real
inference time only a weather forecast (with its own error) would be available, so
the model is trained with more certainty about the future than it will have in
production. This is an accepted limitation of the current data, not a bug.

## Using for inference
1. activate venv
2. run the following: python src/darts_prophet/inference.py --config [relative filepath to your config] --horizon [how many full hours ahead should be predicted, default 336] --offset [how many hours after the end of the training data should the first inference be made, default 0]

## Testing a single model
1. Open src/darts_prophet/single_model_test.py
2. Stel 'CONFIG_PATH' in op de gewenste config.yaml. Het is handig om een kopie te maken van standard_config, 
3. Vervang in 

```
from darts.models import TiDEModel
 
    MODEL_CLS = TiDEModel
```
'TiDEModel' door het gewenste dart model
4. Run the file using the venv.
5. o view tensorboard, open the venv and run "tensorboard --logdir experiments/tb_logs" in terminal.
