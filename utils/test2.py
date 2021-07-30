import os
import signal

def sigint():
    print("sigint")
    exit(1)

signal.signal(signal.SIGINT, sigint)
# signal.signal(signal.SIGKILL, sigkill)
print(os.getcwd())
while True:
    pass