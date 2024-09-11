DOCKER_CMD := docker run -it --rm --gpus=all --privileged=true --ipc=host -v $(shell pwd):/app -v /media/hdd/models/docker:/opt/ml/models
DOCKER_PY_CMD := ${DOCKER_CMD} --entrypoint=python
DOCKER_NSYS_CMD := ${DOCKER_CMD} --entrypoint=nsys
PROFILE_CMD := profile -t cuda,cublas,cudnn,nvtx,osrt --force-overwrite=true --delay=2 --duration=30

PROFILE_TARGETS = logs/tuning_baseline.qdrep logs/tuning_postprocess_1.qdrep

.PHONY: sleep 


build-container: Dockerfile
	docker build -f $< -t video-inference:dev .


run-container: build-container
	${DOCKER_CMD} video-inference:dev


logs/cli.pipeline.dot:
	${DOCKER_CMD} --entrypoint=gst-launch-1.0 video-inference:dev filesrc location=media/in.mp4 num-buffers=200 ! decodebin ! progressreport update-freq=1 ! fakesink sync=true


logs/%.pipeline.dot: %.py
	${DOCKER_PY_CMD} video-inference:dev $<


logs/%.qdrep: %.py
	${DOCKER_NSYS_CMD} video-inference:dev ${PROFILE_CMD} -o $@ python $<


%.pipeline.png: logs/%.pipeline.dot
	dot -Tpng -o$@ $< && rm -f $<


%.output.svg: %.rec
	cat $< | svg-term > $@
	
%.rec:
	asciinema rec $@ -c "$(MAKE) --no-print-directory logs/$*.pipeline.dot sleep"

sleep:
	@sleep 2
	@echo "---"


pipeline: cli.pipeline.png frames_into_python.pipeline.png frames_into_pytorch.pipeline.png

tuning: logs/tuning_baseline.qdrep logs/tuning_postprocess_1.qdrep logs/tuning_postprocess_2.qdrep logs/tuning_batch.qdrep logs/tuning_fp16.qdrep logs/tuning_dtod.qdrep logs/tuning_concurrency.qdrep