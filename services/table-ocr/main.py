import argparse
from robyn import Robyn

app = Robyn(__file__)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app.start(host=args.host, port=args.port)
