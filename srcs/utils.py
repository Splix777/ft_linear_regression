import logging
import traceback

from pathlib import Path


def error_decorator(debug=False):
    # Initialize logger if not already initialized
    if not hasattr(error_decorator, 'logger'):
        # Make the directory for logs if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        logging.basicConfig(filename='logs/error_log.log', level=logging.ERROR)
        error_decorator.logger = logging.getLogger(__name__)

    def decorator(func):
        def wrapper(*args, **kwargs):
            exceptions = (ValueError, TypeError, FileNotFoundError, Exception)
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                exception_name = e.__class__.__name__
                error_message = f"{exception_name} occurred in function '{func.__name__}': {e}"
                if debug:
                    error_message += "\n" + traceback.format_exc()
                    error_decorator.logger.error(error_message)
                else:
                    print(error_message)

        return wrapper

    return decorator
