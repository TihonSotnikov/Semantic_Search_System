import copy
import logging
import click
import sys
from typing import Literal

from uvicorn.config import LOGGING_CONFIG


TRACE_LOG_LEVEL = 5

class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        TRACE_LOG_LEVEL: "blue",
        logging.DEBUG: "cyan",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red",
    }

    def __init__(self, fmt=None, datefmt=None, style: Literal["%", "{", "$"] = "%"):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_colors = sys.stdout is not None and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_colors:
            return super().format(record)

        record_copy = copy.copy(record)
        if record_copy.args:
            def style_arg(val):
                return click.style(str(val), fg=(224, 177, 130), bold=True)

            if isinstance(record_copy.msg, str):
                msg_temp = record_copy.msg
                for spec in ('%d', '%f', '%i'):
                    msg_temp = msg_temp.replace(spec, '%s')
                record_copy.msg = msg_temp

            try:
                if isinstance(record_copy.args, dict):
                    record_copy.args = {k: style_arg(v) for k, v in record_copy.args.items()}
                elif isinstance(record_copy.args, tuple):
                    record_copy.args = tuple(style_arg(v) for v in record_copy.args)
                else:
                    record_copy.args = (style_arg(record_copy.args),)
                record_copy.message = record_copy.getMessage()
            except Exception:
                record_copy = copy.copy(record)

        return super().format(record_copy)

    def formatMessage(self, record: logging.LogRecord) -> str:
        if not self.use_colors:
            return super().formatMessage(record)

        lvl_color = self.LEVEL_COLORS.get(record.levelno, "white")
        is_bold = record.levelno >= logging.ERROR
        record.levelname = click.style(record.levelname, fg=lvl_color, bold=is_bold)
        record.name = click.style(record.name, fg="magenta")

        if hasattr(record, "asctime"):
            record.asctime = click.style(record.asctime, fg="blue")

        return super().formatMessage(record)


def configure_logging(
        logger_name: str,
        output_file: str | None,
        level = logging.INFO
        ) -> None:
    log_format = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(log_format, date_format))

    if output_file:
        file_handler = logging.FileHandler(output_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

    sss_logger = logging.getLogger(logger_name)
    sss_logger.setLevel(level)

    sss_logger.handlers = []
    sss_logger.addHandler(console_handler)
    if output_file:
        sss_logger.addHandler(file_handler)


def get_unified_logging_config(level: str = 'INFO') -> dict:
    """Модифицирует стандартный словарь настроек Uvicorn, внедряя наши обработчики."""
    config = copy.deepcopy(LOGGING_CONFIG)
    
    log_format = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    config["formatters"]["custom_colored"] = {
        "()": "app.logger.logger.ColoredFormatter", 
        "fmt": log_format,
        "datefmt": date_format,
    }
    config["formatters"]["plain"] = {
        "format": log_format,
        "datefmt": date_format,
    }

    config["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "filename": "app.log",
        "mode": "a",
        "encoding": "utf-8",
        "formatter": "plain",
    }

    config["handlers"]["default"]["formatter"] = "custom_colored"
    config["handlers"]["access"]["formatter"] = "custom_colored"

    config["loggers"]["uvicorn"]["handlers"].append("file")
    config["loggers"]["uvicorn.access"]["handlers"].append("file")

    config["loggers"]["semantic_search_system"] = {
        "handlers": ["default", "file"],
        "level": level,
        "propagate": False,
    }

    return config
