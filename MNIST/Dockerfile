FROM continuumio/miniconda3

WORKDIR /app

COPY env.yml /tmp/env.yml

RUN --mount=type=cache,target=/opt/conda/pkgs conda env create -f /tmp/env.yml

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]




RUN echo "which python"
CMD [ "which", "python" ]

COPY train.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "run.py"]