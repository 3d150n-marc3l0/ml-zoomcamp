PIPENV_PYTHON=python3.11 pipenv --python 3.11.5
pipenv install flask joblib scikit-learn==1.5.2

docker image pull svizor/zoomcamp-model:3.11.5-slim

docker build -t zoomcamp-hw05:3.10.12-slim .

docker run -p 5000:5000 zoomcamp-hw05:3.10.12-slim

docker run -d --rm -p 5000:5000 zoomcamp-hw05:3.10.12-slim

docker run -d --rm --name hw05 -p 5000:5000 zoomcamp-hw05:3.10.12-slim




