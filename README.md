# DukeOfTokyo
tweedejaarsproject KI

## Setup
HEAD
### Option A: `uv` (fastest)
1. Install `uv` (macOS):
	- Homebrew: `brew install uv`
	- Or: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create and activate a virtualenv:
	- `uv venv`
	- `source .venv/bin/activate`
3. Install dependencies:
	- `uv pip install -r requirements.txt`

### Option B: built-in `venv` (no extra tools)
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`

### Option C: Conda
1. `conda create -n dukeoftokyo python=3.11`
2. `conda activate dukeoftokyo`
3. `pip install -r requirements.txt`
=======
1. uv venv
2. source .venv/bin/activate
3. uv pip install -r pyproject.toml

Data:
1. Create a folder 'data/'
2. Place the relevant csv files in this folder
3. Edit the config you will use to refer to the correct file (data, data_path)

## Testing a single model
1. Open src/darts_prophet/single_model_test.py
2. Stel 'CONFIG_PATH' in op de gewenste config.yaml
3. Vervang in 

```
from darts.models import TiDEModel
 
    MODEL_CLS = TiDEModel
```
'TiDEModel' door het gewenste dart model
4. Run met de venv het bestand.
origin/checkpoints_2

## Running packages under website: 
```
python -m website.notification.check_threshold
```

## ADD BASE_DIR
```
BASE_DIR = Path(__file__).resolve().parent.parent.parent
```
Add this at the top of each file, so that relative paths from config work. .parent goes one level up.