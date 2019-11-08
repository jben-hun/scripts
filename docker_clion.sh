#! /bin/bash

CONTAINER=${1-bjenei_gpu_0}

docker exec -u $UID:$(id -g $USER) -d ${CONTAINER} /home/bjenei/app/clion-2019.2/bin/clion.sh
