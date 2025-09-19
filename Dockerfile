FROM python:3.10

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl openssl git \
    tesseract-ocr libtesseract-dev poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Add corporate CA(s) in PEM (can contain multiple certs), owned by root
COPY --chown=root:root cacert.pem /usr/local/share/ca-certificates/corp-bundle.crt
RUN chmod 0644 /usr/local/share/ca-certificates/corp-bundle.crt \
    && update-ca-certificates

# (Optional) These are not required if the system bundle is correct.
# ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    TESSERACT_CMD=/usr/bin/tesseract

WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip cache purge \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
