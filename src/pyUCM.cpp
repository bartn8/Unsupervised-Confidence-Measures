#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>

#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>

#include "generate_samples.hpp"
#include "confidence_measures.hpp"
#include "DSI.hpp"
// #include "evaluation.hpp"


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
    _DSI dsi_LL;
    _DSI dsi_RR;
    uint32 bad;
    uint32 width, height;
    uint32 dmin, dmax;
    float32 threshold; 
    const char *choices_positive_ptr, *choices_negative_ptr;
    int log = 0;

    //Uso funzione dissimilarità (Census)
    bool similarity = 0;

    //Tmp
    std::vector<std::string> choices_positive;
    std::vector<std::string> choices_negative;
    std::vector<std::string> choices;    
    std::string choices_positive_str;
    std::string choices_negative_str;
    std::vector<cv::Mat> confidences;
    std::vector<std::string> confidence_names; 
    cv::Mat positive_samples, negative_samples;
    std::time_t start_ms, stop_ms;
    // Mat rgb;

    //Output
    float32 *psamples;
    float32 *nsamples;
    
    //In vars

    PyObject *_leftarg = NULL, *_rightarg = NULL;
    PyObject *_dleftarg = NULL, *_drightarg = NULL;
    PyObject *_dsilrarg = NULL;
    PyObject *_dsirrarg = NULL, *_dsillarg = NULL;

    PyObject *_left = NULL, *_right = NULL;
    PyObject *_dleft = NULL, *_dright = NULL;
    PyObject *_dsilr = NULL;
    PyObject *_dsirr = NULL, *_dsill = NULL;

    //Out vars
    PyObject *_pconfidencesarg = NULL, *_pconfidences = NULL;
    PyObject *_nconfidencesarg = NULL, *_nconfidences = NULL;

    //pconfidences, nconfidences, left, right, displ, dispr, dsilr, dsill, dsirr, bad, width, height, dmin, dmax, threshold, choices_pos, choices_neg

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!IIIIIfss|p", 
        &PyArray_Type, &_pconfidencesarg,
        &PyArray_Type, &_nconfidencesarg,
        &PyArray_Type, &_leftarg,
        &PyArray_Type, &_rightarg,
        &PyArray_Type, &_dleftarg,
        &PyArray_Type, &_drightarg,
        &PyArray_Type, &_dsilrarg,
        &PyArray_Type, &_dsillarg,
        &PyArray_Type, &_dsirrarg,
        &bad, &width, &height, &dmin, &dmax, &threshold, &choices_positive_ptr, &choices_negative_ptr, &log)) return NULL;

    if(dmin >= dmax){
        PyErr_Format(PyExc_TypeError,
                    "Dmin should be < than dmax (%ld < %ld)", dmin, dmax);
        goto fail;
    }
    
    if(threshold < 0){
        PyErr_Format(PyExc_TypeError,
                    "binary threshold should be greater than zero (%f)", threshold);
        goto fail;
    }

    if(bad <= 0){
        PyErr_Format(PyExc_TypeError,
                    "BAD threshold should be greater than zero (%ld)", bad);
        goto fail;
    }

    //cout << width << " " << height << " " << dmin << " " << dmax << endl;

    #if NPY_API_VERSION >= 0x0000000c
        _pconfidences = PyArray_FROM_OTF(_pconfidencesarg, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
    #else
        _pconfidences = PyArray_FROM_OTF(_pconfidencesarg, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY);
    #endif
    if (_pconfidences == NULL) goto fail;

    #if NPY_API_VERSION >= 0x0000000c
        _nconfidences = PyArray_FROM_OTF(_nconfidencesarg, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
    #else
        _nconfidences = PyArray_FROM_OTF(_nconfidencesarg, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY);
    #endif
    if (_nconfidences == NULL) goto fail;

    _left = PyArray_FROM_OTF(_leftarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
    if (_left == NULL) goto fail;

    _right = PyArray_FROM_OTF(_rightarg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
    if (_right == NULL) goto fail;

    _dleft = PyArray_FROM_OTF(_dleftarg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (_dleft == NULL) goto fail;

    _dright = PyArray_FROM_OTF(_drightarg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (_dright == NULL) goto fail;
    
    _dsilr = PyArray_FROM_OTF(_dsilrarg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (_dsilr == NULL) goto fail;

    _dsill = PyArray_FROM_OTF(_dsillarg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (_dsill == NULL) goto fail;   

    _dsirr = PyArray_FROM_OTF(_dsirrarg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (_dsirr == NULL) goto fail;             

    psamples = (float32*)PyArray_DATA(_pconfidences);
    nsamples = (float32*)PyArray_DATA(_nconfidences);

    gray_0 = cv::Mat(height, width, CV_8U, (uint8*) PyArray_DATA(_left));
    gray_1 = cv::Mat(height, width, CV_8U, (uint8*) PyArray_DATA(_right));
    disparity_L2R = cv::Mat(height, width, CV_32F, (float32*) PyArray_DATA(_dleft));
    disparity_R2L = cv::Mat(height, width, CV_32F, (float32*) PyArray_DATA(_dright));
    dsi_LR = DSI_init_frombuffer(height, width, dmin, dmax, similarity, (float32*) PyArray_DATA(_dsilr));
    dsi_LL = DSI_init_frombuffer(height, width, dmin, dmax, similarity, (float32*) PyArray_DATA(_dsill));
    dsi_RR = DSI_init_frombuffer(height, width, dmin, dmax, similarity, (float32*) PyArray_DATA(_dsirr));

    //Choices parsing
    choices_negative_str = std::string(choices_negative_ptr);
    tokenize(choices_negative_str, ' ', choices_negative);

    choices_positive_str = std::string(choices_positive_ptr);
    tokenize(choices_positive_str, ' ', choices_positive);

    std::copy(choices_positive.begin(), choices_positive.end(), std::back_inserter(choices));
    std::copy(choices_negative.begin(), choices_negative.end(), std::back_inserter(choices));
    std::sort(choices.begin(), choices.end());
    choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
    
    start_ms = std::time(nullptr);

    fn_confidence_measure(gray_0, gray_1, disparity_L2R, disparity_R2L, dsi_LR, dsi_LL, dsi_RR, bad, choices, confidence_names, confidences);

    stop_ms = std::time(nullptr);
    
    if(log)
        std::cout << "fn_confidence_measure (s): " << (stop_ms-start_ms) << endl;

    start_ms = std::time(nullptr);

    generate_training_samples(confidences, disparity_L2R, threshold, confidence_names, choices_positive, choices_negative, positive_samples, negative_samples);

    stop_ms = std::time(nullptr);

    if(log)
        std::cout << "generate_training_samples (s): " << (stop_ms-start_ms) << endl;
    
    // samples_on_image(gray_0, positive_samples, negative_samples, rgb);
	// imwrite("myoutput/positive_samples.png", positive_samples);
	// imwrite("myoutput/negative_samples.png", negative_samples);
	// imwrite("myoutput/rgb_samples.png", rgb);
	// write_disparity_map(disparity_L2R, positive_samples, "myoutput/disparity_positive.png");
	// write_disparity_map(disparity_L2R, negative_samples, "myoutput/disparity_negative.png");
	// write_disparity_map(disparity_L2R, Mat(), "myoutput/disparity.png");


    //Pass confidences
    for(uint32 y = 0; y < height; y++){
        for(uint32 x = 0; x < width; x++){
            psamples[y*width+x] = positive_samples.at<float32>(y,x);
            nsamples[y*width+x] = negative_samples.at<float32>(y,x);
        }
    }

    #if NPY_API_VERSION >= 0x0000000c
        PyArray_ResolveWritebackIfCopy((PyArrayObject*)_pconfidences);
    #endif
    Py_DECREF(_pconfidences);

    #if NPY_API_VERSION >= 0x0000000c
        PyArray_ResolveWritebackIfCopy((PyArrayObject*)_nconfidences);
    #endif
    Py_DECREF(_nconfidences);

    Py_DECREF(_left);
    Py_DECREF(_right);
    Py_DECREF(_dleft);
    Py_DECREF(_dright);
    Py_DECREF(_dsilr);
    Py_DECREF(_dsill);
    Py_DECREF(_dsirr);
    
    Py_INCREF(Py_None);
    return Py_None;

    fail:
    #if NPY_API_VERSION >= 0x0000000c
        PyArray_DiscardWritebackIfCopy((PyArrayObject*)_pconfidences);
    #endif
    Py_XDECREF(_pconfidences);
    #if NPY_API_VERSION >= 0x0000000c
        PyArray_DiscardWritebackIfCopy((PyArrayObject*)_nconfidences);
    #endif
    Py_XDECREF(_nconfidences);

    Py_XDECREF(_left);
    Py_XDECREF(_right);
    Py_XDECREF(_dleft);
    Py_XDECREF(_dright);
    Py_XDECREF(_dsilr);
    Py_XDECREF(_dsill);
    Py_XDECREF(_dsirr);

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

