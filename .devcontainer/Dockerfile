
FROM python:3

# Copy default endpoint specific user settings overrides into container to specify Python path
# COPY .devcontainer/settings.vscode.json /root/.vscoe

# Install pylint
RUN pip install pylint

# Install git, process tools
RUN apt-get update && apt-get -y install git procps

# Install any missing dependencies for enhanced language service
RUN apt-get install -y libicu[0-9][0-9]

RUN mkdir /workspace
WORKDIR /workspace

# Install Python dependencies from requirements.txt if it exists
COPY requirements.txt* /workspace/
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt && rm requirements.txt*; fi

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
