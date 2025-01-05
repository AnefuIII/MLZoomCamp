FROM public.ecr.aws/lambda/python:3.9-x86_64

RUN pip install keras-image-helper
RUN pip install tensorflow

COPY capstone1model.tflite .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
