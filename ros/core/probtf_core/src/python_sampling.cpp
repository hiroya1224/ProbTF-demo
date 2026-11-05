#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <probtf_core/bingham_sampler.hpp>

#include <exception>
#include <stdexcept>
#include <string>

namespace {

struct Buffer {
  Py_buffer view{};
  ~Buffer() {
    if (view.obj != nullptr) {
      PyBuffer_Release(&view);
    }
  }
  bool get(PyObject* object, bool writable) {
    return PyObject_GetBuffer(object, &view,
        PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS |
        (writable ? PyBUF_WRITABLE : 0)) == 0;
  }
  bool isMatrix() const {
    return view.ndim == 2 && view.itemsize == sizeof(double) &&
           view.format != nullptr && std::string(view.format) == "d";
  }
};

PyObject* sampleBingham(PyObject*, PyObject* args) {
  PyObject* parameter_object;
  PyObject* output_object;
  PyObject* seed_object;
  if (!PyArg_ParseTuple(args, "OOO:sample_bingham", &parameter_object,
                        &output_object, &seed_object)) {
    return nullptr;
  }
  const unsigned long long seed = PyLong_AsUnsignedLongLong(seed_object);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  Buffer parameter;
  Buffer output;
  if (!parameter.get(parameter_object, false) || !output.get(output_object, true)) {
    return nullptr;
  }
  if (!parameter.isMatrix() || !output.isMatrix() ||
      parameter.view.shape[0] < 2 ||
      parameter.view.shape[0] != parameter.view.shape[1] ||
      output.view.shape[1] != parameter.view.shape[0]) {
    PyErr_SetString(PyExc_ValueError,
        "Expected contiguous float64 parameter (q,q) and writable output (n,q), q >= 2.");
    return nullptr;
  }
  try {
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const Eigen::Map<const RowMatrix> matrix(
        static_cast<const double*>(parameter.view.buf), parameter.view.shape[0],
        parameter.view.shape[1]);
    const auto sampler = probtf_core::BinghamSampler::fromParameterMatrix(matrix);
    std::mt19937_64 engine(seed);
    probtf_core::StandardSamplingRandomSource<std::mt19937_64> random(engine);
    probtf_core::BinghamSamplingStatistics statistics;
    std::exception_ptr failure;
    PyThreadState* state = PyEval_SaveThread();
    try {
      sampler.drawBatch(static_cast<std::size_t>(output.view.shape[0]), random,
                        static_cast<double*>(output.view.buf), &statistics);
    } catch (...) {
      failure = std::current_exception();
    }
    PyEval_RestoreThread(state);
    if (failure) {
      std::rethrow_exception(failure);
    }
    return Py_BuildValue("dK", sampler.proposalScale(),
                         static_cast<unsigned long long>(statistics.proposals));
  } catch (const std::invalid_argument& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  }
  return nullptr;
}

PyMethodDef methods[] = {
    {"sample_bingham", sampleBingham, METH_VARARGS,
     "Draw exp(x' A x) on S^(q-1) using the shared C++ BACG implementation.\n"
     "Arguments: float64 parameter (q,q), writable float64 output (n,q), uint64 seed.\n"
     "Returns (optimal_b, proposal_count)."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_sampling",
    "ROS-independent C++ BACG sampler (Kent et al., arXiv:1310.8110).",
    -1, methods, nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__sampling() { return PyModule_Create(&module); }
