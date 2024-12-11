# Docker version 27.3.1
# Volume (bind mount, read-only) at /datasets
# Volume (named) at /app/latest
FROM ultralytics/ultralytics:8.3.28

WORKDIR /app

COPY src src

CMD [ "python", "src/main.py" ]
