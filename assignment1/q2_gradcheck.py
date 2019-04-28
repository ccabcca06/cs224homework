#!/usr/bin/env python

import numpy as np
import random


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients(f是一个计算损失函数值和梯度的函数)
    x -- the point (numpy array) to check the gradient at（数据，gradcheck方法以该数据值为基础，
    通过梯度的定义方法对梯度的合理取值范围进行计算）
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # np.nditer是numpy array自带的迭代器,flags=['multi_index']表示对a进行多重索引
    # op_flags=['readwrite']表示不仅可以对a进行read（读取），还可以write（写入）
    while not it.finished:
        ix = it.multi_index
        # it.multi_index表示输出元素的索引，形式类似(1,1),(1,2),...,(x,y)...

        # Try modifying x[ix] with h defined above to compute numerical
        # gradients (numgrad).

        # Use the centered difference of the gradient.
        # It has smaller asymptotic error than forward / backward difference
        # methods. If you are curious, check out here:
        # https://math.stackexchange.com/questions/2326181/when-to-use-forward-or-central-difference-approximations

        # Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        x[ix] += h

        random.setstate(rndstate)
        new_f1 = f(x)[0]

        x[ix] -= 2 * h

        random.setstate(rndstate)
        new_f2 = f(x)[0]

        x[ix] += h

        numgrad = (new_f1 - new_f2) / (2 * h)
        # lim h -->0，这里用1e-4近似
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!")


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print("")


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
