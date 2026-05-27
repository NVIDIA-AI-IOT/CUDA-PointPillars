@echo off
cd ../
:: VC free compiler download url:https://aka.ms/vs/17/release/vs_buildtools.exe
echo "====== build windows clientsdk start..."

set ROOT_PATH=D:\win_build_package
:: compile solution
if not "%1"=="" (
    set ROOT_PATH=%1
)
echo "====== ROOT_PATH is :" %ROOT_PATH%
set "CURRENT=%cd%"
cd /d %ROOT_PATH%\BuildTools\VC\Tools\MSVC
for /f "delims=" %%i in ('dir 1* /b') do (
    set MSVC_VER=%%i
    GOTO :BREAK
)
:BREAK
echo "MSVC_VER is "%MSVC_VER%
cd /d "%CURRENT%"
set Path=%ROOT_PATH%\CMake_64\bin;%ROOT_PATH%\BuildTools\VC\Tools\MSVC\%MSVC_VER%\bin\Hostx64\x64;%Path%

::clear bin directory
set BIN_DIR="bin"
if not exist %BIN_DIR% (md %BIN_DIR%) else (del %BIN_DIR% /a/s/q)

call %ROOT_PATH%\BuildTools\VC\Auxiliary\Build\vcvarsall.bat amd64
::create empty build directory
set BUILD_DIR=build_win
if not exist %BUILD_DIR% (md %BUILD_DIR%) else (del %BUILD_DIR% /a/s/q)
cd /d %BUILD_DIR%
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
nmake
nmake install

cd /d "%CURRENT%"
if not exist %BUILD_DIR% (md %BUILD_DIR%) else (del %BUILD_DIR% /a/s/q)
cd /d %BUILD_DIR%
echo "compile sdk_client dynamic library"
cmake -G "NMake Makefiles" -DMAKE_SHARED=ON -DCMAKE_BUILD_TYPE=Release ..
nmake
nmake install

cd /d "%CURRENT%"/apps/tools/py_module
set PIPENV_PIPFILE=%ROOT_PATH%/pyenv/Pipfile
pipenv run python setup.py bdist_wheel
pipenv run python -m mkdocs build


cd /d "%CURRENT%"
if not exist "lib\innolidarsdkclient.lib" (
    echo "====== build inno_clientsdk failed!!!"
    exit
)

echo "====== build windows clientsdk success ^_^"
cd build
