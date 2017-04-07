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
static PyObject *CallbacksError;

/*
 * Define a PyObject reference that holds a callable, i.e., a Python function or method
 * that can be called from C.
 */
static PyObject *my_callback = NULL;

/*
 * Python calls 'set_callback' to provide a Python callable to C. 
 * parameter contains the callable object. 
 */
static PyObject *
set_callback(PyObject *dummy, PyObject *parameter)
{
    PyObject *result = NULL;
    PyObject *temp;

    if (PyArg_ParseTuple(parameter, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        Py_XINCREF(temp);         /* Add a reference to new callback */
        Py_XDECREF(my_callback);  /* Discard reference to previous callback, if any */
        my_callback = temp;       /* Remember new callback */
        /* Boilerplate to return "None", i.e., no error */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}

/*
 * Define a minimal function that calls the callback.
 * This simple example passes no arguments in either direction.
 */
static PyObject *
call_callback_simple(PyObject *dummy1, PyObject *dummy2)
{
    PyObject *result;

    /* Call the Python callable in my_callback */
    result = PyObject_CallObject(my_callback, NULL);
    if (result == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyObject_CallObject failed");
        return NULL; /* Pass error back */
    }
    Py_DECREF(result);
    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * Define a function that calls the callback, and passes arguments.
 * This simple example passes one integer argument in each direction.
 */
static PyObject *
call_callback(PyObject *dummy1, PyObject *parameter)
{
    long arg;
    PyObject *arglist;
    PyObject *result;
    int result_value;

    /* parse the argument passed from Python, and build an argument list to pass back */
    if (!PyArg_ParseTuple(parameter, "l", &arg)) {
        PyErr_SetString(CallbacksError, "PyArg_ParseTuple of parameter failed");
        return NULL;
    }
    /* fprintf(stderr, "C received parameter %ld from Python.\n", arg); */
    arglist = Py_BuildValue("(l)", arg);
    if (arglist == NULL) {
        /* Py_BuildValue failed, perhaps because it ran out of memory */
        PyErr_SetString(CallbacksError, "Py_BuildValue failed");
        return NULL;
    }

    /* Call the Python callable in my_callback */
    result = PyObject_CallObject(my_callback, arglist);
    Py_DECREF(arglist);
    if (result == NULL) {
        PyErr_SetString(CallbacksError, "PyObject_CallObject failed");
        return NULL;
    }
    /* fprintf(stderr, "C received result from Python.\n"); */
    if (PyTuple_Check(result)) {
        if (!PyArg_ParseTuple(result, "i", &result_value)) {
            return NULL;
        }
    }else{
        if (!PyArg_Parse(result, "i", &result_value)) {
            return NULL;
        }
    }
    /* fprintf(stderr, "C received return value %d from Python.\n", result_value); */
    Py_DECREF(result);
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
static PyMethodDef CallbackMethods[] = {
    /* ... */
    {"set_callback", set_callback, METH_VARARGS,
     "Provide a Python callable to C."},
    {"call_callback_simple", call_callback_simple, METH_VARARGS,
     "Have C call the Python callable."},
    {"call_callback", call_callback, METH_VARARGS,
     "Have C call the Python callable."},

    /* ... */
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/*
 * A module definition structure
 */
static struct PyModuleDef callbacks_module = {
   PyModuleDef_HEAD_INIT,
   "callbacks",     /* name of module */
   NULL,            /* pointer to a string containing the module's documentation,
                        or NULL if none is provided */
   -1,              /* size of per-interpreter state of the module,
                        or -1 if the module keeps state in global variables. */
   CallbackMethods  /* the array of PyMethodDef structures describing the module's methods */
};

/*
 * The initialization function for this module. Called by the Python interpreter when the Python
 * program first imports this module. 
 * This function must be named PyInit_name(), where name is the name of this module.
 */
PyMODINIT_FUNC
PyInit_callbacks(void)
{
    PyObject *m;

    /* Create the module, and pass the module definition structure to Python. */
    m = PyModule_Create(&callbacks_module);
    if (m == NULL) {
        return NULL;
    }

    /* Initialize the CallbacksError exception that is unique to this module */
    CallbacksError = PyErr_NewException("callbacks.error", NULL, NULL);
    Py_INCREF(CallbacksError);
    PyModule_AddObject(m, "error", CallbacksError);
    
    return m;
}
