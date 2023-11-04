import logging

def main():
    # Set up logging.
    logger = logging.getLogger('project1_small')
    logger.setLevel(logging.WARN)
    fh = logging.FileHandler('small.log')
    fh.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


if __name__ == '__main__':
    main()