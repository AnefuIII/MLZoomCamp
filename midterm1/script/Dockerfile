FROM python:3.12.7-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_rf_est11_depth15.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]