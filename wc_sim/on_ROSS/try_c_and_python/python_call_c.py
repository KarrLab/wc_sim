'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-06
:Copyright: 2017-2018, Karr Lab
:License: MIT
'''
# example program using example spam module

import spam

for cmd in ["ls -l", "date", "no_such_command", 7, None, 'kill']:
    try:
        status = spam.system(cmd)
        print("'{}' returns: {}".format(cmd, status))
    except Exception as e:
        print("Exception: '{}'".format(str(e)))

print(spam.nothing())
