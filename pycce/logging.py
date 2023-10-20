import logging

def get_logger(name=None, level=logging.INFO):
    """
    Returns a logger instance with the name set to the name of the module that calls this function.
    The logger is configured to write messages to stdout with a specific format.
    """
    if name is not None:
        # Get the name of the calling module
        logger_name = logging.getLogger(name).name
    
        # Create or get a logger instance with the above name
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    # Prevent logging from propagating to the root logger or other loggers
    logger.propagate = False
    
    # If the logger already has handlers, we won't add another one to avoid duplicate logging
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create a console handler that outputs to stdout
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        # Define a formatter and set it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(ch)

    return logger
