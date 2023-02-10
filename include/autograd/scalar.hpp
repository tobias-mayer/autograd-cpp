#ifndef __AUTOGRAD_SCALAR_HPP__
#define __AUTOGRAD_SCALAR_HPP__

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <ostream>
#include <vector>
#include <unordered_set>

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
        vector<ScalarPtr> topologicalOrder;
        std::unordered_set<ScalarPtr> visited;
        build_topo(shared_from_this(), topologicalOrder, visited);
        for (auto scalar : topologicalOrder) {
            scalar->_backward();
        }
    }

    void build_topo(ScalarPtr scalar, vector<ScalarPtr>& topologicalOrder, std::unordered_set<ScalarPtr>& visited) {
        if (visited.find(scalar) != visited.end()) return;

        visited.insert(scalar);
        for (auto child : scalar->children()) {
            build_topo(child, topologicalOrder, visited);
        }

        topologicalOrder.insert(topologicalOrder.begin(), scalar);
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
    auto out = make_shared<Scalar>(
        lhs->data() - rhs->data(),
        vector<ScalarPtr>{lhs, rhs}
    );

    out->_backward = [out, lhs, rhs] {
        lhs->set_grad(lhs->grad() + out->grad());
        rhs->set_grad(rhs->grad() + out->grad());
    };

    return out;
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
