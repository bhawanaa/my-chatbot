FROM python:3.10

# Update package lists and install necessary packages including Tesseract OCR.
RUN apt-get update && apt-get install -y \
    git \
    openssl \
    ca-certificates \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy your CA certificate bundle (make sure it is valid) to the certificates directory.
# If your cacert.pem is in the correct format, you can copy it as a .crt file.
COPY cacert.pem /usr/local/share/ca-certificates/cacert.crt
RUN update-ca-certificates

# Set environment variables to use the updated CA certificates if needed.
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV TESSERACT_CMD=/usr/bin/tesseract

# Create a non-root user.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

EXPOSE 8000
WORKDIR /app

# Copy and install dependencies.
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org

# USER root
# RUN mkdir -p /var/lib/apt/lists/partial && chmod -R 755 /var/lib/apt/lists && \
#     apt-get update && apt-get install -y espeak

# USER root
# RUN mkdir -p /var/lib/apt/lists/partial && chmod -R 755 /var/lib/apt/lists && \
#     apt-get update && apt-get install -y espeak-ng

# USER root
# RUN apt-get update && apt-get install -y ffmpeg


# # Switch back to non-root user for running the application.
# USER user

# Copy the application code.
COPY --chown=user . /app

# Start the application.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
