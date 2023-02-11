from pyautograd import Scalar

a = Scalar(3.0)
b = Scalar(4.0)

z = a * b

z.backward()


