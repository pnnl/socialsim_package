import socialsim as ss
import argparse

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, help='challenge', default='cp4', required=False)
    parser.add_argument('-s', type=str, help='submission filepath', required=True)
    parser.add_argument('-n', type=str, help='nodelist filepath', default=None, required=False)
    args = parser.parse_args()

    submission_filepath = args.s
    challenge = args.c.lower()
    nodelist_filepath = args.n

    validation_flag, validation_report = ss.validation_report(submission_filepath,
                                                              challenge=challenge,
                                                              nodelist_filepath=nodelist_filepath)
    print(validation_report)






