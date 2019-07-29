from arguments import argparser
import time
import os
import sys

args = argparser()
while True:
    print(f'Python Executable: {sys.executable}')
    print(f'Python Version: {sys.version}')
    print(f'Virtualenv: {os.getenv("VIRTUAL_ENV")}')

    time.sleep(1)
