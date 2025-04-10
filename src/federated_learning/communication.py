import struct


def send_msg(sock, msg):
    """Send a Message prefixed with a 4-byte length (network byte order)"""
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    """ Read message length and full message """
    raw_message_len = sock.recv(4)
    if not raw_message_len:
        return None
    msglen = struct.unpack('>I', raw_message_len)[0]
    message_data = bytearray()
    while len(message_data) < msglen:
        packet = sock.recv(msglen - len(message_data))
        if not packet:
            return None
        message_data.extend(packet)
    return message_data