def log(text):
    """Log the text to log.txt"""
    with open("log.txt", 'a') as f:
        f.write(text + '\n')