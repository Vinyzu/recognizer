\# Installation Guide



\## Python Version Support

✅ \*\*Supported\*\*: Python 3.8, 3.9, 3.10, 3.11, 3.12  

❌ \*\*Not supported\*\*: Python 3.13 (dependency conflicts)



\## Quick Installation

\# Installation Guide



\## Python Version Support

✅ \*\*Supported\*\*: Python 3.8, 3.9, 3.10, 3.11, 3.12  

❌ \*\*Not supported\*\*: Python 3.13 (dependency conflicts)



\## Quick Installation

pip install recognizer



text



\## Windows Installation Issues



\### Error: "numpy metadata generation failed"

Fix: Install numpy first

pip install "numpy<2.0.0"

pip install recognizer



text



\### Error: "Unknown compiler(s)"  

Fix: Use binary packages only

pip install --only-binary=all recognizer



text



\### Alternative: Install Build Tools

1\. Download \[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

2\. Install "C++ build tools" workload  

3\. Restart PowerShell

4\. Run: `pip install recognizer`



\## Python 3.13 Users

Switch to Python 3.12:



\*\*Using conda:\*\*

conda create -n recognizer python=3.12

conda activate recognizer

pip install recognizer



text



\*\*Using pyenv (Linux/Mac):\*\*

pyenv install 3.12.0

pyenv local 3.12.0

pip install recognizer



text



\## Verification

python -c "import recognizer; print('✅ Success!')"



text

undefined

