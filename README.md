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

## Testing a single model
1. Open src/darts_prophet/single_model_test.py
2. Stel 'CONFIG_PATH' in op de gewenste config.yaml. Het is handig om een kopie te maken van standard_config, 
3. Vervang in 

```
from darts.models import TiDEModel
 
    MODEL_CLS = TiDEModel
```
'TiDEModel' door het gewenste dart model
4. Run met de venv het bestand.
5. Om tensorboard te zien, activeer in een nieuwe terminal de venv en run "tensorboard --logdir experiments/tb_logs.