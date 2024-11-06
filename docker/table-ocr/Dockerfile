FROM registry.baidubce.com/paddlex/paddlex:paddlex3.0.0b1-paddlepaddle3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6

WORKDIR /app

COPY services/paddle/config .

RUN paddlex --pipeline table_recognition.yaml || true

EXPOSE 8000

CMD ["paddlex", "--pipeline", "table_recognition.yaml", "--serve", "--port", "8000"]