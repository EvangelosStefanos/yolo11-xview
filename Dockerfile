# Docker version 27.3.1
FROM ultralytics/ultralytics:8.3.28

COPY datasets/ /datasets/

WORKDIR /app

COPY src/prepare_xView_data.py src/prepare_xView_data.py

RUN python src/prepare_xView_data.py

COPY config/ config/

COPY src/main.py src/main.py

CMD [ "python", "src/main.py" ]
