sudo xhost local:root

docker run --rm --gpus device=0 -it -v $PWD:/app \
--network=host \
wdnet:v0.1 bash