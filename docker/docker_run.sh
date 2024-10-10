OUT_IMAGE="xuanthanh/d3rl:v0"
WORKSPACE="/mnt/hdd/thanh/workspace"
NAME="thanh_d3rlpy"
DOCKER_HOSTNAME=$(echo $HOSTNAME|cut -c5-)
DOCKER_HOSTNAME="Docker$DOCKER_HOSTNAME"
docker run --gpus all -d -it --name ${NAME} --shm-size 256G --ulimit memlock=-1 \
  --network host --pid host --hostname $DOCKER_HOSTNAME --add-host $DOCKER_HOSTNAME:127.0.0.1\
  --mount type=bind,source=${WORKSPACE},target=/workspace $OUT_IMAGE
docker exec -it ${NAME} bash
