from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import logging
import sys
import six

from builtins import object
from contextlib import contextmanager
from logging import Handler, StreamHandler

# 7-bit C1 ANSI escape sequences
ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


def remove_escape_codes(text):
    """
    Remove ANSI escapes codes from string. It's used to remove
    "colors" from log messages.
    """
    return ANSI_ESCAPE.sub("", text)


class LoggingMixin:
    """
    Convenience super-class to have a logger configured with the class name
    """

    def __init__(self, context=None):
        self._set_context(context)

    @property
    def log(self):
        try:
            return self._log
        except AttributeError:
            self._log = logging.root.getChild(
                self.__class__.__module__ + "." + self.__class__.__name__
            )
            return self._log

    def _set_context(self, context):
        if context is not None:
            set_context(self.log, context)


class StreamLogWriter(object):
    """
    Allows to redirect stdout and stderr to logger
    """

    encoding = None

    def __init__(self, logger, level):
        """
        :param log: The log level method to write to, ie. log.debug, log.warning
        :return:
        """
        self.logger = logger
        self.level = level
        self._buffer = str()

    @property
    def closed(self):
        """
        Returns False to indicate that the stream is not closed (as it will be
        open for the duration of Airflow's lifecycle).

        For compatibility with the io.IOBase interface.
        """
        return False

    def _propagate_log(self, message):
        """
        Propagate message removing escape codes.
        """
        self.logger.log(self.level, remove_escape_codes(message))

    def write(self, message):
        """
        Do whatever it takes to actually log the specified logging record
        :param message: message to log
        """
        if not message.endswith("\n"):
            self._buffer += message
        else:
            self._buffer += message
            self._propagate_log(self._buffer.rstrip())
            self._buffer = str()

    def flush(self):
        """
        Ensure all logging output has been flushed
        """
        if len(self._buffer) > 0:
            self._propagate_log(self._buffer)
            self._buffer = str()

    def isatty(self):
        """
        Returns False to indicate the fd is not connected to a tty(-like) device.
        For compatibility reasons.
        """
        return False


class RedirectStdHandler(StreamHandler):
    """
    This class is like a StreamHandler using sys.stderr/stdout, but always uses
    whatever sys.stderr/stderr is currently set to rather than the value of
    sys.stderr/stdout at handler construction time.
    """

    def __init__(self, stream):
        if not isinstance(stream, six.string_types):
            raise Exception(
                "Cannot use file like objects. Use 'stdout' or 'stderr'"
                " as a str and without 'ext://'."
            )

        self._use_stderr = True
        if "stdout" in stream:
            self._use_stderr = False

        # StreamHandler tries to set self.stream
        Handler.__init__(self)

    @property
    def stream(self):
        if self._use_stderr:
            return sys.stderr

        return sys.stdout


@contextmanager
def redirect_stdout(logger, level):
    writer = StreamLogWriter(logger, level)
    try:
        sys.stdout = writer
        yield
    finally:
        sys.stdout = sys.__stdout__


@contextmanager
def redirect_stderr(logger, level):
    writer = StreamLogWriter(logger, level)
    try:
        sys.stderr = writer
        yield
    finally:
        sys.stderr = sys.__stderr__


def set_context(logger, value):
    """
    Walks the tree of loggers and tries to set the context for each handler
    :param logger: logger
    :param value: value to set
    """
    _logger = logger
    while _logger:
        for handler in _logger.handlers:
            try:
                handler.set_context(value)
            except AttributeError:
                # Not all handlers need to have context passed in so we ignore
                # the error when handlers do not have set_context defined.
                pass
        if _logger.propagate is True:
            _logger = _logger.parent
        else:
            _logger = None
