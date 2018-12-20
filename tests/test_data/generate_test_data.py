import json
import random
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

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def generate_test_data(n, filename, header=None):
    with open(filename,'a+') as file:

        if not header is None:
            file.write(json.dumps(header)+'\n')

        for i in range(n):
            if i%1000==0:
                print(i/n)

            line = {'nodeID'        : randomword(10),
                    'nodeUserID'    : randomword(10),
                    'rootID'        : randomword(10),
                    'parentID'      : randomword(10),
                    'nodeTime'      : randomword(10),
                    'actionType'    : randomword(10),
                    'actionSubType' : randomword(10)}

            file.write(json.dumps(line)+'\n')

if __name__=='__main__':
    header = {'identifier' : 'test_submission',
              'team'       : 'testers',
              'scenario'   : '1',
              'domain'     : 'cve',
              'platform'   : 'github'}

    test_files = {'small_test_submission.json' : 10000,
                  'medium_test_submission.json': 100000,
                  'large_test_submission.json' : 1000000}

    for filename, n in test_files.items():
        generate_test_data(n, filename, header=header)
