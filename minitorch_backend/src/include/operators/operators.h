#pragma once

#include <functional>
#include <vector>

namespace minitorch {

class operators {
public:
    template <typename T>
    static auto id(T x) -> T;

    template <typename T>
    static auto neg(T x) -> T;

    template <typename T>
    static auto log(T x) -> T;

    template <typename T>
    static auto log_back(T x, T d) -> T;

    template <typename T>
    static auto exp(T x) -> T;

    template <typename T>
    static auto inv(T x) -> T;

    template <typename T>
    static auto inv_back(T x, T d) -> T;

    template <typename T>
    static auto mul(T x, T y) -> T;

    template <typename T>
    static auto add(T x, T y) -> T;

    template <typename T>
    static auto lt(T x, T y) -> T;

    template <typename T>
    static auto eq(T x, T y) -> T;

    template <typename T>
    static auto max(T x, T y) -> T;

    template <typename T>
    static auto is_close(T x, T y) -> T;

    template <typename T>
    static auto sigmoid(T x) -> T;

    template <typename T>
    static auto sigmoid_back(T x, T d) -> T;

    template <typename T>
    static auto relu(T x) -> T;

    template <typename T>
    static auto relu_back(T x, T d) -> T;

    template <typename T>
    static auto prod(std::vector<T> x) -> T;
};
}