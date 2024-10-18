import streamlit as st
import time
class TimeIt(object):
    def __init__(self):
        self.prev_time = time.time()

    def tick(self, msg=''):
        new_time = time.time()
        delta = new_time - self.prev_time
        print('_%s [Elapsed time: %0.3fs]_' % (msg, delta))
        self.prev_time = new_time