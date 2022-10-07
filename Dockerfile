FROM python:3.10-slim

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache notebook jupyterlab && \
    pip install -r requirements.txt


# create user with a home directory
# ARG NB_USER
# ARG NB_UID
# ENV USER ${NB_USER}
# ENV HOME /home/${NB_USER}

# RUN adduser \
#     --disabled-password \
#     --gecos "Default user" \
#     --uid ${NB_UID} \
#     ${NB_USER}


# WORKDIR ${HOME}
# USER ${USER}

