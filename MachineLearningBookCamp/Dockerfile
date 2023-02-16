FROM continuumio/conda-ci-linux-64-python3.8:latest

WORKDIR /mlbc

# Update jupyter

RUN pip install jupyter --upgrade

RUN sudo mkdir notebooks

ENV PATH="${PATH}:/home/test_user/.local/lib:/home/test_user/.local/bin"

# Update kaggle cli

RUN pip install kaggle --upgrade

COPY kaggle.json /home/test_user/.kaggle/kaggle.json

# Test kaggle

RUN kaggle datasets list

RUN kaggle datasets download -d CooperUnion/cardataset

# Clone the repo

RUN git clone https://github.com/alexeygrigorev/mlbookcamp-code.git
