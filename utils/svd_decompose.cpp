#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <thread>
#include <Eigen/Dense>
#include <mutex>

namespace py = pybind11;

std::mutex mtx;

void svd_worker(const py::array_t<double>& t, py::array_t<double>& result, int start, int end) {
    auto t_buf = t.request();
    auto result_buf = result.request();

    double* t_ptr = (double*)t_buf.ptr;
    double* result_ptr = (double*)result_buf.ptr;

    int shape1 = t_buf.shape[1];
    int shape2 = t_buf.shape[2];

    for (int i = start; i < end; ++i) {
        Eigen::Map<Eigen::MatrixXd> matrix(t_ptr + i * shape1 * shape2, shape1, shape2);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd singular_values = svd.singularValues();

        std::lock_guard<std::mutex> lock(mtx);
        for (int j = 0; j < shape1; ++j) {
            result_ptr[i * shape1 + j] = singular_values(j);
        }
    }
}

py::array_t<double> svd_decompose(py::array_t<double> t, int n) {
    auto t_buf = t.request();
    int num_tensors = t_buf.shape[0];
    int shape1 = t_buf.shape[1];

    py::array_t<double> result({num_tensors, shape1});

    std::vector<std::thread> threads;
    int chunk_size = num_tensors / n;
    for (int i = 0; i < n; ++i) {
        int start = i * chunk_size;
        int end = (i == n - 1) ? num_tensors : start + chunk_size;
        threads.emplace_back(svd_worker, std::ref(t), std::ref(result), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

PYBIND11_MODULE(svd_decompose, m) {
    m.def("svd_decompose", &svd_decompose, "A function that performs SVD decomposition on tensors.");
}
