container_name=$1

xhost +local:
docker run -it --ipc=host --runtime=nvidia \
  --user=$(id -u) \
  -e DISPLAY=$DISPLAY \
  -e CONTAINER_NAME=cuda \
  -e USER=$USER \
  --workdir=/home/$USER \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "/etc/group:/etc/group:ro" \
  -v "/etc/passwd:/etc/passwd:ro" \
  -v "/etc/shadow:/etc/shadow:ro" \
  -v "/etc/sudoers.d:/etc/sudoers.d:ro" \
  -v "/home/$USER/:/home/$USER/" \
  --device=/dev/dri:/dev/dri \
  --name=${container_name} \
  luoxin0826/semantic_segmentation:latest
