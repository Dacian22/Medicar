# check if connection to ssh server is possible

import paramiko
import sys

def check_ssh_connection(host, port, user, password):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port, user, password)
        ssh.close()
        return True
    except Exception as e:
        print(e)
        return False
    
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: %s <host> <port> <user> <password>' % sys.argv[0])
        sys.exit(1)
    host = sys.argv[1]
    port = int(sys.argv[2])
    user = sys.argv[3]
    password = sys.argv[4]
    if check_ssh_connection(host, port, user, password):
        print('Connection to %s:%d as %s is possible' % (host, port, user))
    else:
        print('Connection to %s:%d as %s is not possible' % (host, port, user))
        sys.exit(1)
    sys.exit(0)
    