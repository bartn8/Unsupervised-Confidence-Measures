#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>

#include <sstream>
#include <iostream>
#include <vector>
#include <string>

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
 
  


    static PyObject *_confidence_measure(PyObject *self, PyObject *args)
    { 
        //Input params
        cv::Mat gray_0;
        const char *pointer;
        std::string instr, outstr;
        PyObject *_returnstr; 

        gray_0 = cv::Mat(10,10,CV_8U,cv::Scalar(-1));

        if (!PyArg_ParseTuple(args, "s", &pointer)) return NULL;

        instr = std::string(pointer);
        outstr = instr + " ";
        outstr = outstr + std::to_string(gray_0.rows);
        
        _returnstr = PyUnicode_DecodeUTF8(outstr.c_str(), outstr.size(),NULL);

        return _returnstr;
    } 

    static PyMethodDef UCMMethods[] = {
        {"confidence_measure", _confidence_measure, METH_VARARGS, "Confidence Measure. Select positive and negative"},
        {NULL, NULL, 0, NULL}
    };


    static struct PyModuleDef pyUCMmodule = {
        PyModuleDef_HEAD_INIT,
        "pyUCM",
        "UCM library",
        -1,
        UCMMethods
    };

    PyMODINIT_FUNC PyInit_pyUCM(void) {
        import_array();
        return PyModule_Create(&pyUCMmodule);
    }


