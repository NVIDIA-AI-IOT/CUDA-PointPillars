import socket, ssl, pprint, time
import random
import struct

default_host = "10.168.1.62" # Change this to your Lidar IP
default_port = 9347 # Change this to your Lidar PTC Port

class PTC:
    def __init__(self, host=default_host, port=default_port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lp = random.randint(10000, 30000)
        self.s.bind(('0.0.0.0', lp))
        self.s.settimeout(30)
        self.s.connect((host, port))

    def closeSocket(self):
        self.s.shutdown(1)
        # self.s.close()

    def ByteToHex(self, h):
        return ''.join(["%02x" % x for x in h]).strip()

    def read_bytes(self, payload_size):
        chunks = []
        bytes_received = 0
        while bytes_received < payload_size:
            chunk = self.s.recv(payload_size - bytes_received)
            if chunk == b"":
                raise RuntimeError("Socket has been unexpectedly closed")
            chunks.append(chunk)
            bytes_received = bytes_received + len(chunk)

        return b"".join(chunks)

    def sender(self, cmd_code, payload):
        """
        :param cmd_code:
        :param payload: payload should input like '015d010207'
        :return:
        """
        if cmd_code in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e',
                        'f']:
            cmd_code = "0" + str(cmd_code)
        print("payload :")
        print(payload)
        if not payload or payload.upper() == "NONE":
            payload_len = 0
            p = '4774' + str(cmd_code) + "00" + struct.pack('>L', payload_len).hex()
        else:
            payload_len = len(bytes.fromhex(payload))
            p = '4774' + str(cmd_code) + "00" + struct.pack('>L', payload_len).hex() + payload
        data = bytes.fromhex(p)
        self.s.send(data)
        response = self.s.recv(8)
        print("response: ")
        print(response)
        r_cmd = bytes.hex(response[2:3])
        r_returnCode = bytes.hex(response[3:4])
        if bytes.hex(response[4:8]) == "\x00\x00\x00\x00":
            r_length = 0
            response_payload = ""
        else:
            r_length = int(bytes.hex(response[4:8]), 16)
            response_payload = self.read_bytes(r_length)
        print("command is: %s, get return code: %s, return length: %s, \nreturn string:\n%s" % (
            r_cmd, r_returnCode, r_length, response_payload))
        final_response = {
            "response_command": r_cmd,
            "response_return_code": r_returnCode,
            "response_payload_length": r_length,
            "response_payload": response_payload
        }
        return final_response

if __name__ == "__main__":
    ss = PTC()
    #print(ss.sender('20', '0a1f0a3f09400000'))
    #print(ss.sender('08', None)) # Example command to 'Get Current Fault Log', change the command code and payload as needed
    print(ss.sender('17', '00C8')) # Example command to 'Get Current Fault Log', change the command code and payload as needed
    ss.closeSocket()