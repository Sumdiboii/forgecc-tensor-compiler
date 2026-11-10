#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace forgecc::test {

struct TensorReference {
    std::vector<std::size_t> shape;
    std::vector<float> data;
};

inline TensorReference deterministicTensor(std::vector<std::size_t> shape) {
    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        count *= dimension;
    }

    TensorReference tensor{std::move(shape), {}};
    tensor.data.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float sign = index % 2 == 0 ? -1.0F : 1.0F;
        tensor.data.push_back(sign * static_cast<float>(index + 1));
    }
    return tensor;
}

inline TensorReference elementwise(const TensorReference& left, const TensorReference& right,
                                  char operation) {
    TensorReference result{left.shape, {}};
    result.data.reserve(left.data.size());
    for (std::size_t index = 0; index < left.data.size(); ++index) {
        switch (operation) {
        case '+':
            result.data.push_back(left.data[index] + right.data[index]);
            break;
        case '-':
            result.data.push_back(left.data[index] - right.data[index]);
            break;
        case '*':
            result.data.push_back(left.data[index] * right.data[index]);
            break;
        case '/':
            result.data.push_back(left.data[index] / right.data[index]);
            break;
        default:
            result.data.push_back(0.0F);
            break;
        }
    }
    return result;
}

inline TensorReference relu(const TensorReference& input) {
    TensorReference result{input.shape, {}};
    result.data.reserve(input.data.size());
    for (const float value : input.data) {
        result.data.push_back(value > 0.0F ? value : 0.0F);
    }
    return result;
}

inline TensorReference matmul(const TensorReference& left, const TensorReference& right) {
    const std::size_t rows = left.shape[0];
    const std::size_t inner = left.shape[1];
    const std::size_t columns = right.shape[1];
    TensorReference result{{rows, columns}, std::vector<float>(rows * columns, 0.0F)};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t shared = 0; shared < inner; ++shared) {
                result.data[row * columns + column] +=
                    left.data[row * inner + shared] * right.data[shared * columns + column];
            }
        }
    }
    return result;
}

inline std::optional<std::vector<float>> parseOutput(const std::string& output) {
    std::istringstream stream(output);
    std::vector<float> values;
    float value = 0.0F;
    while (stream >> value) {
        values.push_back(value);
    }
    if (!stream.eof()) {
        return std::nullopt;
    }
    return values;
}

struct ComparisonResult {
    bool passed = false;
    float maximumAbsoluteError = 0.0F;
    std::size_t mismatchedElements = 0;
    std::size_t comparedElements = 0;
    std::string failure;
};

inline ComparisonResult compare(const std::vector<float>& expected,
                                const std::vector<float>& actual,
                                float tolerance = 1.0e-5F) {
    ComparisonResult result;
    if (expected.size() != actual.size()) {
        result.failure = "expected " + std::to_string(expected.size()) +
                         " values but received " + std::to_string(actual.size());
        return result;
    }

    result.comparedElements = expected.size();
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float absoluteError = std::abs(expected[index] - actual[index]);
        if (std::isfinite(absoluteError)) {
            result.maximumAbsoluteError = std::max(result.maximumAbsoluteError, absoluteError);
        }
        if (!std::isfinite(absoluteError) || absoluteError > tolerance) {
            ++result.mismatchedElements;
            if (result.failure.empty()) {
                std::ostringstream message;
                message << std::setprecision(8) << "element " << index << ": expected "
                        << expected[index] << ", actual " << actual[index]
                        << ", absolute error " << absoluteError;
                result.failure = message.str();
            }
        }
    }
    result.passed = result.mismatchedElements == 0;
    return result;
}

} // namespace forgecc::test
