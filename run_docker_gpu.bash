docker run --rm --gpus device=3 -it -v $PWD:/app \
--network=host \
--shm-size=2G \
wdnet:v0.1 bash