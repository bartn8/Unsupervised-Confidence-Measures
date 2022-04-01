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
    //https://www.techiedelight.com/split-string-cpp-using-delimiter/
    void tokenize(std::string const &str, const char delim,
                std::vector<std::string> &out)
    {
        size_t start;
        size_t end = 0;
    
        while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
        {
            end = str.find(delim, start);
            out.push_back(str.substr(start, end - start));
        }
    }

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
        uint32 bad;
        uint32 width, height;
        uint32 dmin, dmax;
        float32 threshold; 
        char *choices_positive_ptr, *choices_negative_ptr;

        //Uso funzione dissimilarità (Census)
        bool similarity = 0;

        //negative+positive
        std::vector<std::string> choices_positive;
        std::vector<std::string> choices_negative;
        std::vector<std::string> choices;

        //Tmp strings
        std::string choices_positive_str
        std::string choices_negative_str;
        
        //Output
	    std::vector<cv::Mat> confidences;
	    std::vector<std::string> confidence_names; 
        
        //In vars

        PyObject *_leftarg = NULL, *_rightarg = NULL;
        PyObject *_dleftarg = NULL, *_drightarg = NULL;
        PyObject *_dsilrarg = NULL, *_dsirlarg = NULL;
        PyObject *_dsirrarg = NULL, *_dsillarg = NULL;

        PyObject *_left = NULL, *_right = NULL;
        PyObject *_dleft = NULL, *_dright = NULL;
        PyObject *_dsilr = NULL, *_dsirl = NULL;
        PyObject *_dsirr = NULL, *_dsill = NULL;

        //Out vars
        PyObject *_confidencesarg = NULL, *_confidences = NULL;

        //confidences, left, right, displ, dispr, dsilr, dsirl, dsill, dsirr, bad, width, height, dmin, dmax, threshold, choices_pos, choices_neg

        if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!IIIIIIss", 
         &PyArray_Type, &_confidencesarg,
         &PyArray_Type, &_leftarg,
         &PyArray_Type, &_rightarg,
         &PyArray_Type, &_dleftarg,
         &PyArray_Type, &_drightarg,
         &PyArray_Type, &_dsilrarg,
         &PyArray_Type, &_dsirlarg,
         &PyArray_Type, &_dsirrarg,
         &PyArray_Type, &_dsillarg,
          &bad, &width, &height, &dmin, &dmax, &threshold, choices_positive_ptr, choices_negative_ptr)) return NULL;

        
        #if NPY_API_VERSION >= 0x0000000c
            _confidences = PyArray_FROM_OTF(_confidencesarg, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
        #else
            _confidences = PyArray_FROM_OTF(_confidencesarg, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY);
        #endif
        if (_confidences == NULL) goto fail;

        _left = PyArray_FROM_OTF(_leftarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_left == NULL) goto fail;

        _right = PyArray_FROM_OTF(_rightarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_right == NULL) goto fail;

        _dleft = PyArray_FROM_OTF(_dleftarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_dleft == NULL) goto fail;

        _dright = PyArray_FROM_OTF(_drightarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_dright == NULL) goto fail;
        
        _dsilr = PyArray_FROM_OTF(_dsilrarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_dsilr == NULL) goto fail;

        _dsirl = PyArray_FROM_OTF(_dsirlarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_dsirl == NULL) goto fail;

        _dsirr = PyArray_FROM_OTF(_dsirrarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_dsirr == NULL) goto fail;

        _dsill = PyArray_FROM_OTF(_dsillarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_dsill == NULL) goto fail;                

        gray_0 = cv::Mat(height, width, cv::CV_8U, (uint8*) PyArray_DATA(_left));
        gray_1 = cv::Mat(height, width, cv::CV_8U, (uint8*) PyArray_DATA(_left));
        

        //Choices parsing
        choices_negative_str = std::string(choices_negative_ptr);
        tokenize(choices_negative_str, ' ', choices_negative);

        choices_positive_str = std::string(choices_positive_ptr);
        tokenize(choices_positive_str, ' ', choices_positive);

        //Matrix creation

        //Penalizzare i rossi
        //Confidenza binaria o flottante? Test
        

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

