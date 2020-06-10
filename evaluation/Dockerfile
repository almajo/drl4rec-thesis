FROM ubuntu

RUN apt-get update && apt-get install -y --no-install-recommends\
    python3-dev \
    ghostscript \
    vim \
    screen \
    python3-pip \
    htop \
    python3-setuptools \
    texlive-fonts-recommended \
    python3-wheel \
    dvipng \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install wheel numpy scipy pandas matplotlib seaborn tensorboard
RUN pip3 install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /home/stud/grimmalex/thesis