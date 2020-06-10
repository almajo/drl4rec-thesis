FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get install -y --no-install-recommends\
    xvfb \
    libgl1-mesa-dev \
    libharfbuzz-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libassimp-dev \
    htop \
    python3-dev \
    vim \
    screen \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/

RUN pip install -r /workspace/requirements.txt
RUN pip install git+https://github.com/maciejkula/spotlight

RUN python -c "import torch, spotlight, gym"
