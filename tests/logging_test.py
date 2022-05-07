# not a pytest test


import logging

if __name__ == "__main__":
    logfile = 'test.log'
    file_handler = logging.FileHandler(logfile)
    file_handler.set_name('fh')
    stream_handler = logging.StreamHandler()
    stream_handler.set_name('sh')
    logging.basicConfig(
        handlers=[file_handler, stream_handler],
        level=logging.INFO,
        format="%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.info('test with both')
    l = logging.getLogger()
    print(l.handlers)
    for h in l.handlers:
        print(h)
        if isinstance(h, logging.FileHandler):
            h_saved = h
            logging.getLogger().removeHandler(h)
    print(logging.getLogger().handlers)
    logging.info('test after removing file handler')
    logging.getLogger().addHandler(h_saved)
    logging.info('test after re-adding file handler')
