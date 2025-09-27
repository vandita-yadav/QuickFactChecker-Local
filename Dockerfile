# Use a full Python image (not slim)
FROM python:3.12

# Set working directory inside container
WORKDIR /app

# Copy everything from repo into container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install all required Python packages
RUN pip install --no-cache-dir \
    streamlit==1.50.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    matplotlib==3.10.0 \
    seaborn==0.13.2 \
    scikit-learn==1.3.2 \
    lime==0.2.0.1 \
    pyngrok \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Expose the Streamlit port
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
