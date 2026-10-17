@echo off
setlocal EnableDelayedExpansion

echo ==========================================
echo  magli — Build wheel and clean leftovers
echo ==========================================
echo.

echo [1/4] Cleaning old build artefacts...
if exist dist rmdir /s /q dist
call :clean
echo         Done.
echo.

echo [2/4] Building wheel...
python -m build --wheel
if errorlevel 1 (
    echo ERROR: Wheel build failed.
    echo Make sure you have 'build' installed:  pip install build
    exit /b 1
)
echo         Done.
echo.

echo [3/4] Removing leftover build artefacts...
call :clean
echo         Done.
echo.

echo [4/4] Build complete. Wheel(s) in dist\:
if exist dist (
    for %%w in (dist\*.whl) do echo         %%~nxw
) else (
    echo         (no dist folder found)
)
echo.
echo All done.
endlocal
exit /b 0

:clean
if exist build            rmdir /s /q build
if exist magli.egg-info   rmdir /s /q magli.egg-info
if exist src\magli.egg-info rmdir /s /q src\magli.egg-info
for /r %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
for /r %%f in (*.pyc) do if exist "%%f" del /q "%%f" 2>nul
exit /b 0

