PROJECT_NAME=sentiment
CONTEXT=dockercontext

clear:

	docker stop ${PROJECT_NAME}
	docker rm ${PROJECT_NAME}
	docker rmi ${PROJECT_NAME}

build:

	docker stop ${PROJECT_NAME} || true && docker rm ${PROJECT_NAME} || true
	docker build -t ${PROJECT_NAME} .
	docker run -dt -w /workspace  --name ${PROJECT_NAME} -v ${PWD}/../:/workspace -w /workspace -p 8080:8080 -p 8888:8888 -p 6666:6666 --env-file ../.env ${PROJECT_NAME} sleep infinity

enter:

	docker exec -it ${PROJECT_NAME}  /bin/bash

