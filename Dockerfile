FROM python:3.9.16-slim-buster as builder

WORKDIR /app

RUN apt update -y && apt-get update -y
RUN pip install --upgrade pip && pip install poetry 
COPY pyproject.toml poetry.toml README.md ./
COPY openai_helper ./openai_helper
COPY conf ./conf

ENTRYPOINT ["sleep", "10000000"]
# ENTRYPOINT .venv/bin/python3.9 -m openai_helper.test_chat_api_util
