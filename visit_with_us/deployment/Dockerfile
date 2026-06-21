
# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files from current directory
COPY . .

# Install dependencies
RUN pip3 install -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
