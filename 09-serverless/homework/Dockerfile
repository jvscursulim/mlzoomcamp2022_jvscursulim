FROM svizor42/zoomcamp-dino-dragon-lambda:v2

RUN pip install numpy
RUN pip install tensorflow
RUN pip install Pillow

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]