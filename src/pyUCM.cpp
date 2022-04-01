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

#include "DSI.hpp"


extern "C" {
    static PyObject *_confidence_measure(PyObject *self, PyObject *args)
    {
        //Input params
        cv::Mat gray_0;
        cv::Mat gray_1;
        cv::Mat disparity_L2R;
        cv::Mat disparity_R2L;
        _DSI dsi_LR;
        _DSI dsi_RL;
        _DSI dsi_LL;
        _DSI dsi_RR;
        int bad;
        uint32 width, height;
        uint32 dmin, dmax;
        float32 threshold; 
        std::vector<std::string> choices_positive;
        std::vector<std::string> choices_negative;

        //Uso funzione dissimilarità (Census)
        bool similarity = 0;

        //negative+positive
        std::vector<std::string> choices;

        //Output
	    std::vector<cv::Mat> confidences;
	    std::vector<std::string> confidence_names; 
        
        PyObject *_sourcearg=NULL, *_destarg=NULL;
        PyObject *_source=NULL, *_dest=NULL;

        //left, right, displ, dispr, dsilr, dsirl, dsill, dsirr, bad, width, height, dmin, dmax, threshold, choices_pos, choices_neg

        if (!PyArg_ParseTuple(args, "O!O!II", &PyArray_Type, &_sourcearg,
         &PyArray_Type, &_destarg, &width, &height)) return NULL;

        if(width % 16 != 0){
            PyErr_Format(PyExc_TypeError,
                     "Width must be a multiple of 16 (%ldx%ld)", width, height);
            goto fail;
        }

        _source = PyArray_FROM_OTF(_sourcearg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_source == NULL) goto fail;

        //TODO: vedere se necessario o basta NPY_ARRAY_IN_ARRAY
        #if NPY_API_VERSION >= 0x0000000c
            _dest = PyArray_FROM_OTF(_destarg, NPY_UINT, NPY_ARRAY_INOUT_ARRAY2);
        #else
            _dest = PyArray_FROM_OTF(_destarg, NPY_UINT, NPY_ARRAY_INOUT_ARRAY);
        #endif

        if (_dest == NULL) goto fail;

        source = (uint8*) PyArray_DATA(_source);
        dest = (uint32*) PyArray_DATA(_dest);

        //Need another array because memory aligment in SSE is different.
        source_mm = (uint8*)_mm_malloc(width*height*sizeof(uint8), 16);
        dest_mm = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                source_mm[y*width+x] = source[y*width+x];
            }
        }

        census5x5_SSE(source_mm, dest_mm, width, height);

        _mm_free(source_mm);
        
        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                dest[y*width+x] = dest_mm[y*width+x];
            }
        }
        
        _mm_free(dest_mm);

        Py_DECREF(_source);
        
        #if NPY_API_VERSION >= 0x0000000c
            PyArray_ResolveWritebackIfCopy((PyArrayObject*)_dest);
        #endif
        
        Py_DECREF(_dest);
        Py_INCREF(Py_None);
        return Py_None;

        fail:

        Py_XDECREF(_source);

        #if NPY_API_VERSION >= 0x0000000c
            PyArray_DiscardWritebackIfCopy((PyArrayObject*)_dest);
        #endif

        Py_XDECREF(_dest);

        return NULL;
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
}

