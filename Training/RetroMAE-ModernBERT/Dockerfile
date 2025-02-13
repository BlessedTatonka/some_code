FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN apt-get update

WORKDIR /app

RUN pip install wandb
RUN pip install faiss-gpu sentencepiece mteb sentence-transformers
RUN pip install transformers==4.48.1
# Yea, two times. For some reason it works only this way
RUN pip install -U flash_attn --no-build-isolation --force-reinstall
RUN pip install -U flash_attn --no-build-isolation --force-reinstall
RUN pip install -U torchvision
RUN pip install 'accelerate>=0.26.0'
RUN pip install razdel
RUN python3 -m pip install --upgrade 'optree>=0.13.0'

ENTRYPOINT [ "bash" ]
