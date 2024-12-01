# Docker version 27.3.1
FROM ultralytics/ultralytics:8.3.28

COPY datasets/xview_chipped /datasets/xview_chipped/

WORKDIR /app

COPY src src

# COPY outputs/app/latest latest

CMD [ "python", "src/main.py" ]
