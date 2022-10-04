FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 4242

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:4242", "predict:app"]