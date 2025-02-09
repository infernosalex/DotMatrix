# Build stage for frontend
FROM node:18 AS frontend-build
WORKDIR /app/web
COPY web/package*.json ./
RUN npm install
COPY web/ ./
RUN npm run build

# Production stage
FROM python:3.9-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy Python application
COPY python/ ./python/

# Copy built frontend from build stage
COPY --from=frontend-build /app/web/dist ./web/dist

# Set environment variables
ENV FLASK_APP=python/api.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 8000

# Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--chdir", "python", "api:app"] 