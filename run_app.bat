@echo off
echo ======================================================================
echo Launching Task 6: Clinical Trial Screening ^& Inference Benchmark App
echo ======================================================================
echo Starting Flask web server at http://127.0.0.1:5000 ...
start "" http://127.0.0.1:5000
python app/app.py --host 127.0.0.1 --port 5000
pause
