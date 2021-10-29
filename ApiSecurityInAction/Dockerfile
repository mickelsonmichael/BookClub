FROM maven:3-openjdk-11

WORKDIR /src

COPY download_repo.sh ./
COPY docker_start.sh ./
RUN chmod +x ./download_repo.sh && ./download_repo.sh
RUN chmod +x ./docker_start.sh

EXPOSE 4567
ENTRYPOINT [ "/src/docker_start.sh" ]
