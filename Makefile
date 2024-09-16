DOCKER_CMD := docker run -it --rm --gpus=all --privileged=true --ipc=host -v /media/ssd/workspace/video-inference:/app -v /media/hdd/models/docker:/opt/ml/models
DOCKER_NSYS_CMD := ${DOCKER_CMD} --entrypoint=nsys
PROFILE_CMD := profile -t cuda,cublas,cudnn,nvtx,osrt --force-overwrite=true --delay=2 --duration=30

build-container: Dockerfile
	docker build -f $< -t video-inference:dev .


run-container: build-container
	${DOCKER_CMD} video-inference:dev

logs/%.qdrep: %.py
	${DOCKER_NSYS_CMD} video-inference:dev ${PROFILE_CMD} -o $@ python3 $<


