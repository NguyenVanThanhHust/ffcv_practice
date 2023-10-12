# ffcv_practice

## Installation
Build docker
```
docker build -t ffcv_img .
```

Create docker container
```
docker run -it --gpus all --network host --name ffcv_ctn -v ${PWD}:/workspace/ ffcv_img
```