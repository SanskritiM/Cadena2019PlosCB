 Set-ExecutionPolicy -ExecutionPolicy Unrestricted

python -m venv .research1

.research1/Scripts/activate.ps1

pip install build ipykernel

pip install numpy tqdm scipy "tensorflow<=2.12" scikit-image datajoint tf-slim 

# https://setuptools.pypa.io/en/latest/userguide/quickstart.html

python -m build

