import logging
import os.path as path
import os

# TODO(Arthur): make configurable
LOGGING_ROOT_DIR=path.expanduser( "~/tmp/Sequential_WC_Simulator_logging")
# make the dir if it doesn't exist
# from http://stackoverflow.com/a/14364249/509882
try: 
    os.makedirs(LOGGING_ROOT_DIR)
except OSError:
    if not path.isdir(LOGGING_ROOT_DIR):
        raise

# from http://stackoverflow.com/a/17037016/509882
def setup_logger(logger_name, log_file=None, level=logging.INFO, to_stdout=False):
    """create and config a logger"""
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(name)s : %(pathname)s:%(lineno)d #%(message)s')
    if not log_file:
        log_file = logger_name + '.log'
    log_file = path.join( LOGGING_ROOT_DIR, log_file )
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    
    if to_stdout:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.addHandler(streamHandler)    

