@echo off
setlocal EnableDelayedExpansion

echo ==========================================
echo  linGAM — Build wheel and clean leftovers
echo ==========================================
echo.

:: Step 1 — clean old build artefacts
echo [1/4] Cleaning old build artefacts...
if exist build          rmdir /s /q build
if exist dist           rmdir /s /q dist
if exist linGAM.egg-info  rmdir /s /q linGAM.egg-info
if exist src\linGAM.egg-info  rmdir /s /q src\linGAM.egg-info
for /r %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
for /r %%f in (*.pyc) do if exist "%%f" del /q "%%f" 2>nul
echo         Done.
echo.

:: Step 2 — build wheel
echo [2/4] Building wheel...
python -m build --wheel
if errorlevel 1 (
    echo ERROR: Wheel build failed.
    echo Make sure you have 'build' installed:  pip install build
    exit /b 1
)
echo         Done.
echo.

:: Step 3 — delete build artefacts but keep the wheel
echo [3/4] Removing leftover build artefacts...
if exist build          rmdir /s /q build
if exist linGAM.egg-info  rmdir /s /q linGAM.egg-info
if exist src\linGAM.egg-info  rmdir /s /q src\linGAM.egg-info
for /r %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
for /r %%f in (*.pyc) do if exist "%%f" del /q "%%f" 2>nul
echo         Done.
echo.

:: Step 4 — report
echo [4/4] Build complete. Wheel(s) in dist\:
if exist dist (
    for %%w in (dist\*.whl) do echo         %%~nxw
) else (
    echo         (no dist folder found)
)
echo.
echo All done.
endlocal
