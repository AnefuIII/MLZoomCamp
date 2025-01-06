FROM public.ecr.aws/lambda/python:3.9-x86_64

WORKDIR /var/task

VOLUME ["/var/task/CIFAKE"]

RUN pip install keras-image-helper tensorflow
#RUN pip install tensorflow

COPY capstone1model.tflite .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
