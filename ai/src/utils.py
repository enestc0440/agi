import logging
import psutil
import gc
import jax

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def memory_guard(max_mem=4.5):
    process = psutil.Process()
    while True:
        if process.memory_info().rss / 1e9 > max_mem:
            gc.collect()
            jax.clear_backends()