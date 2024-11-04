FROM registry.baidubce.com/paddlex/paddlex:paddlex3.0.0b1-paddlepaddle3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6

WORKDIR /app

COPY services/paddle/config .

EXPOSE 8000

CMD ["paddlex", "--pipeline", "OCR.yaml", "--serve", "--port", "8000"]