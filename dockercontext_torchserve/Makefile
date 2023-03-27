PROJECT_NAME=torchserve_sentiment

clear:

	docker stop ${PROJECT_NAME}
	docker rm ${PROJECT_NAME}
	docker rmi ${PROJECT_NAME}

build:

	docker stop ${PROJECT_NAME} || true && docker rm ${PROJECT_NAME} || true
	docker build -t ${PROJECT_NAME} .
	docker run -dt --name ${PROJECT_NAME} -p 9090:8080 -p 9091:8081 -p 9092:8082 ${PROJECT_NAME} 

enter:

	docker exec -it ${PROJECT_NAME}  /bin/bash

