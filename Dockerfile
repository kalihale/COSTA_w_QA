# Use Ubuntu 22.04
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y libpq-dev gcc
RUN pip3 install
RUN pip3 install torch torchvision torchaudio torchtext tensorboard --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install transformers
RUN python3 -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
RUN python3 training.py