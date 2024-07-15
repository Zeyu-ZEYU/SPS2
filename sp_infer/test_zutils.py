import argparse
import time

import torch

import zutils.net as znet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--client", action="store_true")
    args = parser.parse_args()

    if args.client:
        conn = znet.SocketMsger.tcp_connect("10.155.48.72", 48147)
        tensor = torch.ones([2048, 1, 50504], dtype=torch.float16)
        print(tensor.shape)
        conn.send(tensor)
    else:
        listener = znet.SocketMsger.tcp_listener("10.155.48.72", 48147)
        conn, _ = listener.accept()
        data = conn.recv()
        print(data.shape)

    time.sleep(9)
