import logging
import logging.config

logging.config.fileConfig('logging.conf')
log_ = logging.getLogger()
log_.warning("This is root logger, Warning!")