'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-06
:Copyright: 2017, Karr Lab
:License: MIT
'''
# example program using example callbacks module
import callbacks

def f1():
    print('Python executing f1')

print('\nPython calling set_callback(f1)')
callbacks.set_callback(f1)
print('Python calling call_callback_simple() in C')
rv = callbacks.call_callback_simple()
print('call_callback_simple() returns {}'.format(rv))

def f2(n):
    print('Python function f2 received parameter {} from C'.format(n))
    print('Python function f2 returns {} to C'.format(1+n))
    return 1+n,

print('\nPython calling set_callback(f2)')
callbacks.set_callback(f2)
print('Python calling call_callback(3) in C')
rv = callbacks.call_callback(3)
print('call_callback() returns {}'.format(rv))

def f3(n):
    print('Python function f3 received parameter {} from C'.format(n))
    print('Python function f3 returns {} to C'.format(1+n))
    return (1+n,)

print('\nPython calling set_callback(f3)')
callbacks.set_callback(f3)
print('Python calling call_callback(3) in C')
rv = callbacks.call_callback(3)
print('call_callback() returns {}'.format(rv))

