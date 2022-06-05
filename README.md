# andrey-source
HW ML in prod


скачать и запустить контейнер 
https://hub.docker.com/repository/docker/andreykuzmenko2907/hw2
~~~
docker pull andreykuzmenko2907/hw2:v1
docker run -p 8000:8000 andreykuzmenko2907/hw2:v1
~~~
сделать прогноз (нужно находиться в директории online_inference)

~~~
python make_request.py
~~~

