FROM python:3.9

# Create a non-root user
RUN useradd -m -u 1000 user

# Set up directories with correct permissions
RUN mkdir -p /app/data/uploads /app/data/db /.streamlit /.cache && \
    chown -R user:user /app /app/data /app/data/uploads /app/data/db /.streamlit /.cache

# Switch to non-root user
USER user

WORKDIR /app

# Set environment variables
ENV TRANSFORMERS_CACHE="/.cache"
ENV HF_HOME="/.cache"
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy requirements first for better caching
COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . .

# Create a shell script to run both services
RUN echo '#!/bin/bash\n\
# Start the backend\n\
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 & \n\
\n\
# Set environment variables\n\
export BACKEND_API_URL="http://localhost:8000"\n\
\n\
# Start the frontend\n\
python -m streamlit run app.py --server.port=7860 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose both ports
EXPOSE 8000 7860

# Command to run both services
CMD ["/app/start.sh"] 