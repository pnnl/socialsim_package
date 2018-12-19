import datetime

class RecordKeeper:
    def __init__(self, log_file='evaluation_log.txt'):
        self.log_file = log_file
        self.timesheet = {}

    def update(self, message):
        """
        Description: Writes the given message to the log file.

        Inputs:
            :message: (str) A string to be written to the log file.

        Outputs:
            None

        """
        log_line  = datetime.datetime.now().strftime('%H:%M:%S')
        log_line += ' | '
        log_line += message + '\n'

        with open(self.log_file, 'a') as f:
            f.write(log_line)

    def tic(self, index):
        """
        Description: Starts a timer with the given index

        Inputs:
            :index: (int) The timer number.

        Outputs:
            None

        """
        self.timesheet[index] = [datetime.datetime.now(), None, None]

        return None

    def toc(self, index):
        """
        Description: Stops the timer with the given index and returns the time
                     since it started.

        Inputs:
            :index: (int) The timer number.

        Outputs:
            :index: delta_time (float) Timer in seconds since the timer started.

        """
        end_time   = datetime.datetime.now()
        start_time = self.timesheet[index][0]
        delta_time = end_time - start_time

        self.timesheet[index] = [start_time, end_time, delta_time]

        return delta_time
