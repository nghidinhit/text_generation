import logging


class LoggerSingleton:
    _instances = {}

    def __new__(cls, log_name: str):
        if log_name not in cls._instances:
            instance = super(LoggerSingleton, cls).__new__(cls)
            instance.logger = cls._create_logger(log_name)
            cls._instances[log_name] = instance
        return cls._instances[log_name]

    @staticmethod
    def _create_logger(log_name: str):
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
