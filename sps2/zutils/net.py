import enum
import pickle
import socket
import sys
import threading
import time

import zutils.os as zos


class SocketMsger:
    def __init__(self, socket, is_listener=False):
        self.piggyback_data = None
        self.__socket = socket
        self.__is_listener = is_listener
        self.__is_blocking = True
        self.__recv_buffer = b""
        self.__closed = False

    @property
    def socket(self):
        return self.__socket

    @property
    def is_listener(self):
        return self.__is_listener

    @property
    def is_blocking(self):
        return self.__is_blocking

    @property
    def closed(self):
        if getattr(self.__socket, "_closed") is True and self.__closed is False:
            self.__closed = True
        return self.__closed

    def send(self, data):
        if self.__closed or self.__is_listener:
            return
        if isinstance(data, str):
            data_type = 0
            byte_data = data.encode()
        elif isinstance(data, bytes):
            data_type = 1
            byte_data = data
        else:
            data_type = 2
            byte_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        data_length = len(byte_data)
        self.__socket.sendall(f"META({data_type},{data_length})".encode() + byte_data)

    def recv(self, blocking=True):
        if self.__closed or self.__is_listener:
            return
        if blocking:
            if not self.__is_blocking:
                self.__socket.setblocking(True)
                self.__is_blocking = True
        else:
            if self.__is_blocking:
                self.__socket.setblocking(False)
                self.__is_blocking = False
        index = self.__recv_buffer.find(b"META(")
        while index == -1:
            try:
                data = self.__socket.recv(4096)
                if data == b"":
                    self.__closed = True
                    return
                self.__recv_buffer += data
                index = self.__recv_buffer.find(b"META(")
            except BlockingIOError:
                return
        meta_lindex = index + 5
        index = self.__recv_buffer.find(b")", meta_lindex)
        while index == -1:
            try:
                data = self.__socket.recv(4096)
                if data == b"":
                    self.__closed = True
                    return
                self.__recv_buffer += data
                index = self.__recv_buffer.find(b")", meta_lindex)
            except BlockingIOError:
                return
        meta_rindex = index
        meta = self.__recv_buffer[meta_lindex:meta_rindex].split(b",")
        data_type = int(meta[0])
        data_length = int(meta[1])
        body_lindex = meta_rindex + 1
        while len(self.__recv_buffer) - body_lindex < data_length:
            try:
                data = self.__socket.recv(4096)
                if data == b"":
                    self.__closed = True
                    return
                self.__recv_buffer += data
            except BlockingIOError:
                return
        body_rindex = body_lindex + data_length
        recvd_data = self.__recv_buffer[body_lindex:body_rindex]
        self.__recv_buffer = self.__recv_buffer[body_rindex:]
        if data_type == 0:
            return recvd_data.decode()
        elif data_type == 1:
            return recvd_data
        else:
            return pickle.loads(recvd_data)

    def close(self):
        self.__socket.close()
        self.__closed = True

    @staticmethod
    def tcp_listener(listening_ip, listening_port, backlog=100):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listening_ip, listening_port))
        listener.listen(backlog)
        return SocketMsger(listener, True)

    def accept(self):
        if self.__is_listener:
            conn, address = self.__socket.accept()
            connm = SocketMsger(conn)
            return connm, address

    @staticmethod
    def tcp_connect(ip, port, retry=True):
        sock = None
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((ip, port))
                return SocketMsger(sock)
            except Exception as e:
                print("NOT CONNECTED:", e, file=sys.stderr)
                if not retry:
                    return
                time.sleep(1)


class RemoteProgramStatus(enum.Enum):
    COMPLETED = 0
    ERROR = 1
    NON_EXISTENT = 2
    RUNNING = 3


class RemoteProgramChecker:
    def __init__(self, ip, port, handler):
        self.__ip = ip
        self.__port = port
        self.__handler = handler

    def check_status(self):
        connm = SocketMsger.tcp_connect(self.__ip, self.__port)
        connm.send(("HDL", self.__handler))
        status = connm.recv()
        return status


class RemoteProgramRunner:
    def __init__(self, listening_ip, listening_port):
        self.__listener = SocketMsger.tcp_listener(listening_ip, listening_port)
        self.__thread = threading.Thread(target=self.__listener_thread)
        self.__handler_lock = threading.Lock()
        self.__next_handler = 0
        self.__handler_status = {}
        self.__is_started = False

    def start(self):
        if not self.__is_started:
            self.__is_started = True
            self.__thread.start()

    def join(self):
        self.__thread.join()

    def __listener_thread(self):
        while True:
            connm, _ = self.__listener.accept()
            thread = threading.Thread(target=self.__connm_thread, args=(connm,))
            thread.start()

    def __connm_thread(self, connm):
        request = connm.recv()
        if request is None:
            return
        req_type = request[0]
        req_data = request[1]
        if req_type == "CMD":
            handler = self.__assign_handler()
            self.__handler_status[handler] = RemoteProgramStatus.RUNNING
            connm.send(handler)
            output = zos.run_cmd(req_data, return_output=False)
            if output is None:
                self.__handler_status[handler] = RemoteProgramStatus.ERROR
            else:
                self.__handler_status.pop(handler)
        elif req_type == "HDL":
            if req_data in self.__handler_status:
                connm.send(self.__handler_status[req_data])
            elif req_data >= 0 and req_data < self.__next_handler:
                connm.send(RemoteProgramStatus.COMPLETED)
            else:
                connm.send(RemoteProgramStatus.NON_EXISTENT)

    def __assign_handler(self):
        with self.__handler_lock:
            handler = self.__next_handler
            self.__next_handler += 1
        return handler

    @staticmethod
    def send_cmd(ip, port, cmd):
        connm = SocketMsger.tcp_connect(ip, port)
        connm.send(("CMD", cmd))
        handler = connm.recv()
        return RemoteProgramChecker(ip, port, handler)
