import datetime

class RecordKeeper:
    def __init__(self, log_file='simulation_log.txt'):
        """
        Description:

        Inputs:

        Outputs:

        """
        self.log_file = log_file

    def update(self, message):
        """
        Description:

        Inputs:

        Outputs:

        """
        log_line  = datetime.datetime.now().strftime('%H:%M:%S')
        log_line += ' | '
        log_line += message + '\n'

        with open(self.log_file, 'a') as f:
            f.write(log_line)
