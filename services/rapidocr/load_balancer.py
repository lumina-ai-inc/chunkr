import asyncio
import requests
from flask import Flask, request, Response
import random
import subprocess

BASE_PORT = 8000
NUM_WORKERS = 20

app = Flask(__name__)

def proxy_request():
    worker_port = random.randint(BASE_PORT, BASE_PORT + NUM_WORKERS - 1)
    worker_url = f"http://localhost:{worker_port}/ocr"

    client_request = request.get_data()
    client_headers = {key: value for (key, value) in request.headers if key != 'Host'}

    resp = requests.post(worker_url, data=client_request, headers=client_headers, stream=True)
    return Response(resp.iter_content(chunk_size=10*1024),
                    content_type=resp.headers['Content-Type'],
                    status=resp.status_code)

def start_workers():
    for i in range(NUM_WORKERS):
        port = BASE_PORT + i
        cmd = f"python main.py {port}"
        subprocess.Popen(cmd, shell=True)
        print(f"Started worker on port {port}")

@app.route('/ocr', methods=['POST'])
def handle_request():
    return proxy_request()

if __name__ == "__main__":
    start_workers()
    print("Load balancer running on http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000)