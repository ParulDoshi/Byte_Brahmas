"""
Standalone viewer for the NetworkStreamer output. Run this on the
"mission control" machine (or in a second terminal on the same
laptop, for a demo) to watch the live feed being pushed from main.py
or main_cli.py.

Usage:
    python streaming/receiver_demo.py --port 9000

Then set stream_ip in config/app_config.yaml to this machine's IP
address (or 127.0.0.1 if running both on the same machine) and start
the prototype.
"""

import argparse
import socket
import struct

import cv2
import numpy as np


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Waiting for the prototype to connect on {args.host}:{args.port} ...")
    conn, addr = server.accept()
    print(f"Connected: {addr}")

    try:
        while True:
            header = recv_exact(conn, 4)
            if header is None:
                break
            (length,) = struct.unpack(">I", header)
            data = recv_exact(conn, length)
            if data is None:
                break
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("Mission control - live feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        conn.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
