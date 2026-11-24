@echo off
rem feng run              -> program dosya yolunu sorar
rem feng run dosya.mat    -> dosyayi dogrudan analiz eder
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%predict_file.py"

if /i not "%~1"=="run" goto usage

if "%~2"=="" (
    "%PY%" "%SCRIPT%"
) else (
    "%PY%" "%SCRIPT%" "%~2"
)
exit /b %errorlevel%

:usage
echo Kullanim:
echo   feng run                 dosya yolunu sorar
echo   feng run dosya.mat       dosyayi dogrudan analiz eder
exit /b 1
