import logging
import sys

logger = logging.getLogger("geneal")


def configure_logger(libraries_level=None):
    """
    Makes the "geneal" logger print to stdout, unless the application already
    configured logging, in which case geneal's records go through that
    configuration. The root logger and its handlers are never modified.
    """

    if not isinstance(libraries_level, list):
        libraries_level = []

    for library, level in libraries_level:
        logging.getLogger(library).setLevel(getattr(logging, level))

    if not logger.handlers and not logging.getLogger().handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
