@echo off
REM SpaceMed AI demo runner (Windows)
pip install -r requirements.txt
python -m spacemed.build_index
streamlit run app.py
