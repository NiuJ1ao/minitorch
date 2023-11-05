#include <vector>
#include <pybind11/numpy.h>

namespace minitorch {

namespace py = pybind11;

const size_t ALIGNMENT = 128;

template <typename T>
class NDArray {
public:
    NDArray(const size_t size);
    NDArray(const size_t size, T val);
    NDArray(const std::vector<T> &vec);
    virtual ~NDArray() { free(data); };
    auto operator[](size_t idx) -> T&;
    void assign_slice(const T val);
    auto size() const -> size_t { return size_; };
    void from_numpy(const py::array_t<T> &input_array);
    auto to_numpy() -> py::array_t<T>;

private:
    size_t size_;
    T *data;
};

auto broadcast_index(NDArray<int> &big_index, NDArray<int> &big_shape, NDArray<int> &shape, NDArray<int> &out_index);
auto to_index(size_t ordinal, NDArray<int> &shape, NDArray<int> &out_index);
auto index_to_position(NDArray<int> &index, NDArray<int> &strides) -> size_t;
auto shape_broadcast(NDArray<int> &a, NDArray<int> &b) -> NDArray<int>;

template<typename T>
void declare_array(py::module &m, const std::string &typestr);
void pybind_ndarray(py::module &m);
}

// -GNinja "-DCMAKE_MAKE_PROGRAM:FILEPATH=/home/ycniu/anaconda3/envs/minitorch/lib/python3.8/site-packages/ninja/data/bin/ninja"
// -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/ycniu/minitorch/minitorch_backend/build/lib.linux-x86_64-cpython-38/ -DPYTHON_EXECUTABLE=/home/ycniu/anaconda3/envs/minitorch/bin/python