import logging
import os


def create_logger(folder, name=None):
    """Create a logger to save logs."""
    if not (name is None):
        logname = os.path.join(folder, name)
    else:
        compt = 0
        while os.path.exists(os.path.join(folder,f"logs_{compt}.txt")):
            compt+=1
        logname = os.path.join(folder,f"logs_{compt}.txt")
    
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(logname, mode="w")
    consoleHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    return logger 
    

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """Learning rate policy used in nnUNet."""
    return initial_lr * (1 - epoch / max_epochs)**exponent


def infinite_iterable(i):
    while True:
        yield from i

    
