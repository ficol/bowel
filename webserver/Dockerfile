FROM continuumio/miniconda3

COPY environment.yml .

RUN conda env create -f environment.yml

WORKDIR /bowel

COPY ./bowel /bowel/bowel

COPY ./webserver /bowel/webserver

EXPOSE 8080

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bowel", "python", "-m", "webserver.server"]