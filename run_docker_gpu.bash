docker run --rm --gpus device=3 -it -v $PWD:/app \
--network=host \
--shm-size=32G \
wdnet:v0 bash