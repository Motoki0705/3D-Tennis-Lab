# Use the official Python image as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system packages required for machine learning and development
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# # Install node.js LTS version
# # Fetch the NodeSource GPG key and add the repository, then install nodejs
# RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
#     && apt-get install -y nodejs

# # Install gemini-cli globally
# RUN npm install -g https://github.com/google-gemini/gemini-cli
# # Install claude-cli globally
# RUN npm install -g @anthropic-ai/claude-code

# Install the project dependencies
RUN pip install "poetry==1.8.2"

# copy the poetry files first to leverage Docker cache
COPY poetry.lock pyproject.toml /app/
# Install the dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

EXPOSE 8888
CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]