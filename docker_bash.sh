#! /bin/bash

CONTAINER=${1-bjenei_gpu_0}

docker exec -u $UID:$(id -g $USER) -it ${CONTAINER} /bin/bash
