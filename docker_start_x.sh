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
    -e DISPLAY=$DISPLAY \
    --env="QT_X11_NO_MITSHM=1" \
    --privileged \
    -v /dev/video0:/dev/video0 \
    -v /dev/video1:/dev/video1 \
    -v /dev/video2:/dev/video2 \
    --name $NAME \
    -v ${HOME}:${HOME} \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /fastdata:/fastdata \
    --net host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY \
    -p 0.0.0.0:6006:6006 \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    $IMAGE

PASSWD_ENTRY="$(getent passwd $USER)"
GROUP_ENTRY="$(getent group $(groups))"

docker exec -u 0:0 $NAME sh -c "echo \"$PASSWD_ENTRY\" >> /etc/passwd; echo \"$GROUP_ENTRY\" >> /etc/group"

#docker exec -u 0:0 "${NAME}_gpu${GPU}" sh -c "apt-get install -y gdb"
