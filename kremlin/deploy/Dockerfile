FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
RUN pip install --no-cache-dir '.[witness]' && useradd --system --home /data witness && mkdir -p /data && chown witness /data
USER witness
VOLUME /data
EXPOSE 8443 8081
ENTRYPOINT ["kremlin-witness", "--key", "/data/witness.key", "--receipts", "/data/receipts", "--control", "0.0.0.0:8081"]
