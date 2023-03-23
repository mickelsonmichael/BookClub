FROM continuumio/conda-ci-linux-64-python3.8:latest

WORKDIR /mlbc

# Update PIP packages

COPY requirements.txt .

RUN pip install -r requirements.txt --upgrade

ENV PATH="${PATH}:/home/test_user/.local/lib:/home/test_user/.local/bin"

WORKDIR /mlbc/notebooks

RUN alias launch-notebooks='jupyter notebook --no-browser --ip=0.0.0.0 --notebook-dir=/mlbc/notebooks'

# Update kaggle cli

RUN pip install kaggle --upgrade

COPY kaggle.json /home/test_user/.kaggle/kaggle.json

# Test kaggle

RUN kaggle datasets list

# Clone the repo

RUN git clone https://github.com/alexeygrigorev/mlbookcamp-code.git

# ==== CHAPTER 6 ====

WORKDIR /mlbc

RUN wget https://raw.githubusercontent.com/gastonstat/CreditScoring/master/CreditScoring.csv

RUN sudo chown -R test_user:test_user ~/.local/share/jupyter 

EXPOSE 8888

ENTRYPOINT [ "/bin/bash", "-c" ]
CMD [ "jupyter notebook --no-browser --ip=0.0.0.0 --notebook-dir=/mlbc/notebooks --allow-root" ]
