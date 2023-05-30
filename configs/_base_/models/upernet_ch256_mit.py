# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
_base_ = ['upernet_mit.py']

model = dict(decode_head=dict(channels=256, ))
