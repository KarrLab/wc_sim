/*
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-06
:Copyright: 2017, Karr Lab
:License: MIT
*/

/* Include functions, macros, globals and type definitions for accessing the Python run-time. */
#include <Python.h>

/*
 * Define a Python object (PyObject) reference to an error that can be passed back to Python.
 * Could also use a predeclared C objects corresponding to any built-in Python exception.
 */
static PyObject *SpamError;

/*
 * Define a function that will execute the system() command for a Python spam.system(args) call.
 * args is a Python tuple object containing the arguments.
 * PyArg_ParseTuple parses the string argument args into command.
 * If system() returns an error, PyErr_SetString writes a Python error into SpamError.
 * If system() succeeds, PyLong_FromLong creates a Python long, which is returned to Python.
 */
static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0) {
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }
    return PyLong_FromLong(sts);
}

/*
 * Define a minimal function that does nothing. 
 * Py_None is the C name for the special Python object None.
 */
static PyObject *
spam_do_nothing(PyObject *self, PyObject *args)
{
    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * Define an array of methods supported by this C extension. 
 * Each entry lists
 *   the method name in Python
 *   the function that method maps to in C
 *   the type of argument passing for this method
 *   a string description of the method
 */
static PyMethodDef SpamMethods[] = {
    /* ... */
    {"system", spam_system, METH_VARARGS,
     "Execute a shell command."},
    {"nothing", spam_do_nothing, METH_VARARGS,
     "Do nothing, and return None."},
    /* ... */
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/*
 * A module definition structure
 */
static struct PyModuleDef spammodule = {
   PyModuleDef_HEAD_INIT,
   "spam",      /* name of module */
   NULL,        /* pointer to a string containing the module's documentation, or NULL if none is provided */
   -1,          /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
   SpamMethods  /* the array of PyMethodDef structures describing the module's methods */
};

/*
 * The initialization function for this module. Called by the Python interpreter when the Python
 * program first imports this module, spam. 
 * This function must be named PyInit_name(), where name is the name of this module.
 */
PyMODINIT_FUNC
PyInit_spam(void)
{
    PyObject *m;

    /* Create the module, and pass the module definition structure to Python. */
    m = PyModule_Create(&spammodule);
    if (m == NULL)
        return NULL;

    /* Initialize the SpamError exception that is unique to this module */
    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_INCREF(SpamError);
    PyModule_AddObject(m, "SpamError", SpamError);
    
    return m;
}
