docker run --rm --gpus device=3 -it -v $PWD:/app \
--network=host \
wdnet:v0.1 bash