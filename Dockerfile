# Possible to use a specific version of Python
FROM python:latest
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# copy the content of the local src directory to the working directory (defined above)
COPY . .
# CMD ["python", "app.py"]