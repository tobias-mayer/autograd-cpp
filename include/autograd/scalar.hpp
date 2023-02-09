#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <ostream>
#include <vector>

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
    std::function<void()> _backward;

public:
    Scalar(double data) : _data(data), _grad(0.0f), _backward{[]() {}} { }

    Scalar(double data, vector<ScalarPtr> children)
        : _data(data),
          _grad(0.0f),
          _children{children},
          _backward{[]() {}} { }


    double data() const { return _data; }
    void set_data(double data) { _data = data; }
    double grad() const { return _grad; }
    const vector<ScalarPtr>& children() const { return _children; }
    void set_grad(double grad) { _grad = grad; }

    ValuePtr pow(ValuePtr rhs) {
        auto out = make_shared<Scalar>(
            std::pow(_data, rhs->_data),
            vector<ValuePtr>(shared_from_this())
        );

        return out;
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

    return out;
}

inline ScalarPtr operator-(ScalarPtr lhs, ScalarPtr rhs) {
    auto out = make_shared<Scalar>(
        lhs->data() - rhs->data(),
        vector<ScalarPtr>{lhs, rhs}
    );

    return out;
}

inline ScalarPtr operator*(ScalarPtr lhs, ScalarPtr rhs) {
    auto out = make_shared<Scalar>(
        lhs->data() * rhs->data(),
        vector<ScalarPtr>{lhs, rhs}
    );

    return out;
}

inline ScalarPtr operator/(ScalarPtr lhs, ScalarPtr rhs) {
    return lhs * rhs->pow(make_shared<Scalar>(-1));
}

};

