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
2. Stel 'CONFIG_PATH' in op de gewenste config.yaml
3. Vervang in 

```
from darts.models import TiDEModel
 
    MODEL_CLS = TiDEModel
```
'TiDEModel' door het gewenste dart model
4. Run met de venv het bestand.

## To open the dashboard in streamlit
1. Activate your virtual environment
2. pip install streamlit
3. go to DukeOfTokyo/website
4. streamlit run app.py
5. To stop: Control + C

## To get the incoming reservations:
1. inside website/incoming_reservations create a new folder .streamlit
2. create website/incoming_reservation/.streamlit/secrets.toml
3. Inside secrets.toml insert the API Keys in the format to be read inside read_incoming_data.py
4. Check that .streamlit/secrets.toml is inside .gitignore