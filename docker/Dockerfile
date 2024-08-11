FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

# Timezone setting
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get upgrade -y && \
    apt-get install -y tzdata
ENV TZ Asia/Tokyo

# Install packages
RUN apt-get install -y git vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*
RUN pip install ipykernel \
    pydot pydotplus graphviz \
    tensorflow_datasets tensorflow_text

# Add a non-root user (overwrite by Dev Container)
ARG USER=dev
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID dev \
    && useradd -m -u $UID -g $GID dev

# Create a new home directory (used while the startup of Dev Container)
ARG HOME_DIR=/home/dev
RUN mkdir -p $HOME_DIR
RUN chown -R $UID:$GID $HOME_DIR

# Create directres for VSCode extensions to avoid reinstall extensions
# (ref: https://qiita.com/P-man_Brown/items/645c350ff89d991f6717)
RUN mkdir -p /home/$USER/.vscode-server/extensions \
    /home/$USER/.vscode-server-insiders/extensions \
    && chown -R $USER \
    /home/$USER/.vscode-server \
    /home/$USER/.vscode-server-insiders

USER $USER
ENV HOME $HOME_DIR

WORKDIR /workspace
