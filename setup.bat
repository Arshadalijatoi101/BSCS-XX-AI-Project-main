@echo off
echo ========================================
echo AI Lab Project Setup
echo ========================================

echo Step 1: Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo Step 2: Installing packages...
pip install --no-cache-dir pandas numpy scikit-learn matplotlib seaborn Pillow openpyxl

echo Step 3: Verifying installations...
python -c "import pandas; print('✅ pandas:', pandas.__version__)"
python -c "import numpy; print('✅ numpy:', numpy.__version__)"
python -c "import sklearn; print('✅ scikit-learn:', sklearn.__version__)"
python -c "import matplotlib; print('✅ matplotlib:', matplotlib.__version__)"
python -c "import seaborn; print('✅ seaborn:', seaborn.__version__)"

echo ========================================
echo Setup complete!
echo Now run:
echo   cd src
echo   python main.py
echo ========================================
pause