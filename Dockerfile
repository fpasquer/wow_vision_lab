
FROM python:3.15.2-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

ARG HOST_UID=1000
ARG HOST_GID=1000

RUN if ! [ "${HOST_UID}" -gt 0 ]; then \
        echo "HOST_UID must be a positive, non-root UID." >&2; \
        exit 1; \
    fi \
    && groupadd --gid "${HOST_GID}" app \
    && useradd --uid "${HOST_UID}" --gid app --create-home app

COPY src/ ./src/

USER app

CMD ["python", "--version"]
