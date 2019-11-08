#! /bin/bash

GPU=${1-0}
NAME=${2-bjenei}
#IMAGE=${3-ultinous/camguru:build-CGU-4213-1}
#IMAGE=${3-ultinous/camguru:build-CGU-5555-2}
#IMAGE=${3-ultinous/camguru:build-CGU-5680-1}
IMAGE=${3-ultinous/camguru:build-CGU-6137-2}

NAME=${NAME}_gpu_${GPU//,/_}

docker pull $IMAGE
NV_GPU=$GPU nvidia-docker run --rm -t -d \
    --name $NAME \
    -v ${HOME}:${HOME} \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /fastdata:/fastdata \
    --net host \
    -p 0.0.0.0:6006:6006 \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    $IMAGE

PASSWD_ENTRY="$(getent passwd $USER)"
GROUP_ENTRY="$(getent group $(groups))"

docker exec -u 0:0 $NAME sh -c "echo \"$PASSWD_ENTRY\" >> /etc/passwd; echo \"$GROUP_ENTRY\" >> /etc/group"

#docker exec -u 0:0 "${NAME}_gpu${GPU}" sh -c "apt-get install -y gdb"
