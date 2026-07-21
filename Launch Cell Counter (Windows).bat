@echo off
REM Double-click in Explorer to launch the napari Cell Counter plugin.
REM Works on Windows 10 and 11 with Miniconda/Anaconda installed.

setlocal
cd /d "%~dp0"
set "ENV_NAME=cell-counting"

echo Cell Counter launcher
echo Repo: %CD%
echo.

REM --- conda already on PATH? --------------------------------------------
where conda >nul 2>nul
if %ERRORLEVEL%==0 (
    call conda activate %ENV_NAME% 2>nul
    if not errorlevel 1 goto run
)

REM --- try common install locations --------------------------------------
for %%B in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\mambaforge"
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\Continuum\miniconda3"
    "%LOCALAPPDATA%\Continuum\anaconda3"
    "%ProgramData%\miniconda3"
    "%ProgramData%\Anaconda3"
) do (
    if exist "%%~B\Scripts\activate.bat" (
        call "%%~B\Scripts\activate.bat" %ENV_NAME%
        if not errorlevel 1 goto run
    )
)

echo Could not find conda, or the '%ENV_NAME%' environment does not exist.
echo Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
echo then create the environment:
echo     conda env create -f "%~dp0environment.yml"
echo.
pause
exit /b 1

:run
echo Activated conda env: %ENV_NAME%
echo Starting napari...
where napari-cell-counter >nul 2>nul
if %ERRORLEVEL%==0 (
    napari-cell-counter
) else (
    python -m napari_cell_counter
)

if errorlevel 1 (
    echo.
    echo napari exited with an error.
    pause
)
endlocal
