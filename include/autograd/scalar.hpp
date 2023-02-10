#ifndef __AUTOGRAD_SCALAR_HPP__
#define __AUTOGRAD_SCALAR_HPP__

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <ostream>
#include <vector>
#include <unordered_set>
#include <queue>

namespace autograd {

using std::make_shared;
using std::ostream;
using std::shared_ptr;
using std::vector;

struct Scalar;
using ScalarPtr = shared_ptr<Scalar>;

struct Scalar : public std::enable_shared_from_this<Scalar> {
private:
    float _data;
    double _grad;
    vector<ScalarPtr> _children;

public:
    Scalar(double data) : _data(data), _grad(0.0f), _backward{[]() {}} { }

    Scalar(double data, vector<ScalarPtr> children)
        : _data(data),
          _grad(0.0f),
          _children{children},
          _backward{[]() {}} { }

    std::function<void()> _backward;
    double data() const { return _data; }
    void set_data(double data) { _data = data; }
    double grad() const { return _grad; }
    const vector<ScalarPtr>& children() const { return _children; }
    void set_grad(double grad) { _grad = grad; }

    ScalarPtr pow(ScalarPtr rhs) {
        auto out = make_shared<Scalar>(
            std::pow(_data, rhs->_data),
            vector<ScalarPtr>{ shared_from_this() }
        );

        return out;
    }

    void backward() {
        _grad = 1.0;

        std::queue<ScalarPtr> queue;
        queue.push(shared_from_this());

        std::unordered_set<ScalarPtr> visited;

        while (!queue.empty()) {
            auto scalar = queue.front();
            queue.pop();

            scalar->_backward();
            visited.insert(scalar);

            for (auto child : scalar->children()) {
                if (visited.find(child) != visited.end()) continue;

                queue.push(child);
            }
        }

    }

    friend ostream& operator<<(ostream& os, const Scalar& scalar) {
        os << "Scalar(data=" << scalar._data << ", grad=" << scalar._grad << ")";
        return os;
    }
};

inline ScalarPtr operator+(ScalarPtr lhs, ScalarPtr rhs) {
    auto out = make_shared<Scalar>(
        lhs->data() + rhs->data(),
        vector<ScalarPtr>{lhs, rhs}
    );

    out->_backward = [out, lhs, rhs]() {
        lhs->set_grad(lhs->grad() + out->grad());
        rhs->set_grad(rhs->grad() + out->grad());
    };

    return out;
}

inline ScalarPtr operator-(ScalarPtr lhs, ScalarPtr rhs) {
    return lhs + (-rhs);
}

inline ScalarPtr operator*(ScalarPtr lhs, ScalarPtr rhs) {
    auto out = make_shared<Scalar>(
        lhs->data() * rhs->data(),
        vector<ScalarPtr>{lhs, rhs}
    );

    out->_backward = [out, lhs, rhs] {
        lhs->set_grad(lhs->grad() + rhs->data() * out->grad());
        rhs->set_grad(rhs->grad() + lhs->data() * out->grad());
    };

    return out;
}

inline ScalarPtr operator/(ScalarPtr lhs, ScalarPtr rhs) {
    return lhs * rhs->pow(make_shared<Scalar>(-1));
}

};

#endif
