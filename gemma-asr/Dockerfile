FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    wyoming \
    httpx

# Copy application files
COPY gemma_handler.py .
COPY gemma_wrapper.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10305

ENTRYPOINT ["/app/entrypoint.sh"]
