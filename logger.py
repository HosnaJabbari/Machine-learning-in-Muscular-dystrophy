class Logger:

    log_file = None

    def __init__(self, path):
        if Logger.log_file is None:
            Logger.log_file = open(path, 'w')
        else:
            print "A log file is already created named: ", Logger.log_file

    def log(self, message):
        Logger.log_file.write(message + '\n')
        Logger.log_file.flush()

    def __del__(self):
        if Logger.log_file is not None:
            Logger.log_file.flush()
            Logger.log_file.close()
