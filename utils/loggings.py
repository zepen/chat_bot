# -*- coding: utf-8 -*-
"""
日志模块用于抓取日志
"""
import os
import time
import logging
import inspect
from datetime import datetime

NOW_TIME = datetime.now().strftime("%Y-%m-%d")
LOG_PATH = "logs/"

if os.path.exists(LOG_PATH) is False:
    os.mkdir(LOG_PATH)

handlers = {logging.INFO: LOG_PATH + NOW_TIME + "_info.log",
            logging.WARN: LOG_PATH + NOW_TIME + "_warn.log",
            logging.ERROR: LOG_PATH + NOW_TIME + "_error.log"}


class LOG(object):

    def __init__(self):
        self.__loggers = {}
        self.time_fmt = '%Y-%m-%d %H:%M:%S'
        log_levels = handlers.keys()
        for level in log_levels:
            path = os.path.abspath(handlers[level])
            # noinspection PyTypeChecker
            handlers[level] = logging.FileHandler(path)
        log_levels = handlers.keys()
        for level in log_levels:
            logger = logging.getLogger(str(level))
            logger.addHandler(handlers[level])
            logger.setLevel(level)
            self.__loggers.update({level: logger})

    def print_now_time(self):
        return time.strftime(self.time_fmt, time.localtime())

    def get_log_message(self, level, message):
        frame, filename, line_no, function_name, code, unknow_field = inspect.stack()[2]
        '''log_format：[time] [type] [mark] message'''
        return "[%s] [%s] [%s - %s - %s] %s" % \
               (self.print_now_time(), level, filename, line_no, function_name, message)

    def info(self, message):
        message = self.get_log_message("info", message)
        self.__loggers[logging.INFO].info(message)

    def warn(self, message):
        message = self.get_log_message("warning", message)
        self.__loggers[logging.WARNING].error(message)

    def error(self, message):
        message = self.get_log_message("error", message)
        self.__loggers[logging.ERROR].error(message)


# 实例化日志类
log = LOG()
