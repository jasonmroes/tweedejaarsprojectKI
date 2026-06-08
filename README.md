# DukeOfTokyo
tweedejaarsproject KI

## Setup
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