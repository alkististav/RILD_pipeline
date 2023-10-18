FROM debian:buster-slim
ENV PATH="/opt/dcm2niix-master/bin:$PATH"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           ca-certificates \
           cmake \
           g++ \
           gcc \
           git \
           make \
           pigz \
           zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/rordenlab/dcm2niix /tmp/dcm2niix \
    && mkdir /tmp/dcm2niix/build \
    && cd /tmp/dcm2niix/build \
    && cmake  -DCMAKE_INSTALL_PREFIX:PATH=/opt/dcm2niix-master .. \
    && make -j1 \
    && make install \
    && rm -rf /tmp/dcm2niix
ENV PATH="/opt/niftyreg-master/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/niftyreg-master/lib:$LD_LIBRARY_PATH"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           ca-certificates \
           cmake \
           g++ \
           gcc \
           git \
           make \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/niftyreg/build \
    && git clone https://github.com/KCL-BMEIS/niftyreg /tmp/niftyreg/source \
    && cd /tmp/niftyreg/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/niftyreg-master -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF /tmp/niftyreg/source \
    && make -j1 \
    && make install \
    && ldconfig \
    && rm -rf /tmp/niftyreg
ENV CONDA_DIR="/opt/miniconda-latest" \
    PATH="/opt/miniconda-latest/bin:$PATH"
COPY env.yaml ./env.yaml
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bzip2 \
           ca-certificates \
           curl \
	   ffmpeg \
	   libsm6 \
           libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    # Install dependencies.
    && export PATH="/opt/miniconda-latest/bin:$PATH" \
    && echo "Downloading Miniconda installer ..." \
    && conda_installer="/tmp/miniconda.sh" \
    && curl -fsSL -o "$conda_installer" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash "$conda_installer" -b -p /opt/miniconda-latest \
    && rm -f "$conda_installer" \
    && conda update -yq -nbase conda \
    # Prefer packages in conda-forge
    && conda config --system --prepend channels conda-forge \
    # Packages in lower-priority channels not considered if a package with the same
    # name exists in a higher priority channel. Can dramatically speed up installations.
    # Conda recommends this as a default
    # https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html
    && conda config --set channel_priority strict \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    # Enable `conda activate 
    && conda init bash \
    && conda install -y mamba -c conda-forge 
RUN mamba env update --file env.yaml \
    # Clean up
    && sync && conda clean --all --yes -tipy && sync \
    && rm -rf ~/.cache/pip/*
RUN test "$(getent passwd rild_user)" \
    || useradd --no-user-group --create-home --shell /bin/bash rild_user
USER rild_user
