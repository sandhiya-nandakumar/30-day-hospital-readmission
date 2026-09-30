@echo off
python -m pip install -r requirements.txt
python download_dataset.py
python train.py
streamlit run app.py
