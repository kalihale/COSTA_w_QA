# Use Ubuntu 22.04
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y libpq-dev gcc
RUN apt-get install -y pip
RUN pip3 install torch torchvision torchaudio torchtext tensorboard
RUN pip3 install transformers datasets

WORKDIR /app

COPY . /app

CMD ["bash"]