import json
import random
import time
import string

"""
TODO:
    - Establish standard time format and include that in the test data.
    - parentIDs from previous nodes
    - rootIDs from previous nodes
    - fixed number of action types
    - fixed number of action subtypes for each action type
    -

"""

def random_word(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def random_datetime():

    start  = "1/1/2008 1:30 PM"
    end    = "1/1/2009 4:50 AM"
    format = '%m/%d/%Y %I:%M %p'
    prop   = random.random()
    stime  = time.mktime(time.strptime(start, format))
    etime  = time.mktime(time.strptime(end, format))
    ptime  = stime + prop * (etime - stime)

    datetime = time.strftime(format, time.localtime(ptime))

    return datetime

def random_platform():
    platforms = ['reddit', 'twitter', 'github', 'telegram']

    platform = random.choice(platforms)

    return platform

def generate_test_data(n, filename, header=None):
    with open(filename,'a+') as file:

        if header is not None:
            file.write(json.dumps(header)+'\n')

        for i in range(n):
            if i%1000==0:
                print(i/n)

            line = {'nodeID'        : random_word(10),
                    'nodeUserID'    : random_word(10),
                    'rootID'        : random_word(10),
                    'parentID'      : random_word(10),
                    'nodeTime'      : random_datetime(),
                    'actionType'    : random_word(10),
                    'actionSubType' : random_word(10),
                    'platform'      : random_platform()}

            file.write(json.dumps(line)+'\n')

if __name__=='__main__':
    header = {'identifier' : 'test_submission',
              'team'       : 'testers',
              'scenario'   : '1'}

    test_files = {'ground_truth.json' : 100000,
                  'simulation.json'   : 100000}

    for filename, n in test_files.items():
        generate_test_data(n, filename, header=header)
