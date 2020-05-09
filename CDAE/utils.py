import logging
from os import path


def logger_setup(config: dict):
    dir_root = config["log_root"]
    log_name = config["log_name"]
    full_path = path.join(dir_root, log_name)
    if config["append_time"]:
        from time import strftime, localtime

        full_path += strftime("-%m-%d|%H:%M:%S", localtime())
    full_path += ".log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s| %(message)s", "%m-%d|%H:%M:%S")

    file_hdl = logging.FileHandler(full_path)
    file_hdl.setFormatter(formatter)

    root_logger.addHandler(file_hdl)

    if config["console_output"]:
        console_hdl = logging.StreamHandler()
        console_hdl.setFormatter(formatter)
        root_logger.addHandler(console_hdl)
