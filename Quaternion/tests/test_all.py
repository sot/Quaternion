# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest

from .. import Quat

ra = 10.
dec = 20.
roll = 30.
q0 = Quat([ra, dec, roll])

equatorial_23 = np.array([[[ 10,  20,  30],
                           [ 10,  20, -30],
                           [ 10, -90,  30]],
                          [[ 10,  20,   0],
                           [ 10,  90,  30],
                           [ 10,  90, -30]]])

q_23 = np.array([[[ 0.26853582, -0.14487813,  0.12767944,  0.94371436],
                  [-0.23929834, -0.18930786,  0.03813458,  0.95154852],
                  [ 0.1227878 ,  0.69636424, -0.1227878 ,  0.69636424]],
                 [[ 0.01513444, -0.17298739,  0.08583165,  0.98106026],
                  [ 0.24184476, -0.66446302,  0.24184476,  0.66446302],
                  [ 0.1227878 ,  0.69636424,  0.1227878 , -0.69636424]]])

transform_23 = np.array([[[[  9.25416578e-01,  -3.18795778e-01,  -2.04874129e-01],
                           [  1.63175911e-01,   8.23172945e-01,  -5.43838142e-01],
                           [  3.42020143e-01,   4.69846310e-01,   8.13797681e-01]],
                          [[  9.25416578e-01,   1.80283112e-02,  -3.78522306e-01],
                           [  1.63175911e-01,   8.82564119e-01,   4.40969611e-01],
                           [  3.42020143e-01,  -4.69846310e-01,   8.13797681e-01]],
                          [[  6.03020831e-17,   3.42020143e-01,   9.39692621e-01],
                           [  1.06328842e-17,   9.39692621e-01,  -3.42020143e-01],
                           [ -1.00000000e+00,   3.06161700e-17,   5.30287619e-17]]],
                         [[[  9.25416578e-01,  -1.73648178e-01,  -3.36824089e-01],
                           [  1.63175911e-01,   9.84807753e-01,  -5.93911746e-02],
                           [  3.42020143e-01,   0.00000000e+00,   9.39692621e-01]],
                          [[  6.03020831e-17,  -6.42787610e-01,  -7.66044443e-01],
                           [  1.06328842e-17,   7.66044443e-01,  -6.42787610e-01],
                           [  1.00000000e+00,   3.06161700e-17,   5.30287619e-17]],
                          [[  6.03020831e-17,   3.42020143e-01,  -9.39692621e-01],
                           [  1.06328842e-17,   9.39692621e-01,   3.42020143e-01],
                           [  1.00000000e+00,  -3.06161700e-17,   5.30287619e-17]]]])


def test_init_exceptions():
    with pytest.raises(TypeError):
        q = Quat(np.zeros((2,)))
    with pytest.raises(TypeError):
        q = Quat(np.zeros((5,)))
    with pytest.raises(TypeError):
        q = Quat(equatorial_23)
    with pytest.raises(TypeError):
        q = Quat(q_23)
    with pytest.raises(TypeError):
        q = Quat(transform_23)


def test_from_eq():
    q = Quat([ra, dec, roll])
    assert np.allclose(q.q[0], 0.26853582)
    assert np.allclose(q.q[1], -0.14487813)
    assert np.allclose(q.q[2],  0.12767944)
    assert np.allclose(q.q[3],  0.94371436)
    assert np.allclose(q.roll0, 30)
    assert np.allclose(q.ra0, 10)


def test_from_eq_vectorized():
    # the following line would give unexpected results
    # because  the input is interpreted as a (non-vectorized) transform
    # the shape of the input is (3,3)
    # q = Quat(equatorial_23[0])

    # this is the proper way:
    q = Quat(equatorial=equatorial_23[0])
    assert q.q.shape == (3, 4)
    assert np.allclose(q.q, q_23[0])

    q = Quat(equatorial=equatorial_23)
    assert q.q.shape == (2, 3, 4)
    assert np.allclose(q.q, q_23)


def test_transform_from_eq():
    q = Quat(equatorial=equatorial_23)
    assert q.transform.shape == (2, 3, 3, 3)
    assert np.allclose(q.transform, transform_23)


def test_from_transform():
    """Initialize from inverse of q0 via transform matrix"""
    q = Quat(q0.transform.transpose())
    assert np.allclose(q.q[0], -0.26853582)
    assert np.allclose(q.q[1], 0.14487813)
    assert np.allclose(q.q[2], -0.12767944)
    assert np.allclose(q.q[3],  0.94371436)

    q = Quat(q0.transform)
    assert np.allclose(q.roll0, 30)
    assert np.allclose(q.ra0, 10)


def test_from_transform_vectorized():
    q = Quat(transform=transform_23)
    assert q.q.shape == (2, 3, 4)
    assert np.allclose(q.q, q_23)


def test_eq_from_transform():
    # this raises 'Unexpected negative norm' exception due to roundoff in copy/paste above
    #q = Quat(transform=transform_23)
    #assert q.equatorial.shape == (2, 3, 3)
    #assert np.allclose(q.equatorial, equatorial_23)

    # this one fails (quaternion -> equatorial -> quaternion is not an identity)
    #q = Quat(transform=np.vstack([q0.transform[np.newaxis], q0.transform[np.newaxis]]))
    #assert np.allclose(q.roll0, 30)
    #assert np.allclose(q.ra0, 10)

    t = np.zeros((4,5,3,3))
    t[:] = q0.transform[np.newaxis][np.newaxis]
    q = Quat(transform=t)
    print('roll', q.roll0)
    assert np.allclose(q.roll0, 30)
    assert np.allclose(q.ra0, 10)

    assert q.equatorial.shape == (4, 5, 3)


def test_inv_eq():
    q = Quat(q0.equatorial)
    t = q.transform
    tinv = q.inv().transform
    t_tinv = np.dot(t, tinv)
    for v1, v2 in zip(t_tinv.flatten(), [1, 0, 0, 0, 1, 0, 0, 0, 1]):
        assert np.allclose(v1, v2)


def test_inv_q():
    q = Quat(q0.q)
    t = q.transform
    tinv = q.inv().transform
    t_tinv = np.dot(t, tinv)
    for v1, v2 in zip(t_tinv.flatten(), [1, 0, 0, 0, 1, 0, 0, 0, 1]):
        assert np.allclose(v1, v2)


def test_dq():
    q1 = Quat((20, 30, 0))
    q2 = Quat((20, 30.1, 1))
    dq = q1.dq(q2)
    assert np.allclose(dq.equatorial, (0, 0.1, 1))


def test_ra0_roll0():
    q = Quat(Quat([-1, 0, -2]).q)
    assert np.allclose(q.ra, 359)
    assert np.allclose(q.ra0, -1)
    assert np.allclose(q.roll, 358)
    assert np.allclose(q.roll0, -2)


def test_repr():
    q = Quat([1, 2, 3])
    assert repr(q) == '<Quat q1=0.02632421 q2=-0.01721736 q3=0.00917905 q4=0.99946303>'

    class SubQuat(Quat):
        pass

    q = SubQuat([1, 2, 3])
    assert repr(q) == '<SubQuat q1=0.02632421 q2=-0.01721736 q3=0.00917905 q4=0.99946303>'


def test_numeric_underflow():
    """
    Test new code (below) for numeric issue https://github.com/sot/Quaternion/issues/1.
    If this code is not included then the test fails with a MathDomainError::

        one_minus_xn2 = 1 - xn**2
        if one_minus_xn2 < 0:
            if one_minus_xn2 < -1e-12:
                raise ValueError('Unexpected negative norm: {}'.format(one_minus_xn2))
            one_minus_xn2 = 0
    """
    quat = Quat((0, 0, 0))
    angle = 0
    while angle < 360:
        q = Quat((0, angle, 0))
        quat = q * quat
        quat.equatorial
        angle += 0.1


def test_div_mult():
    q1 = Quat((1, 2, 3))
    q2 = Quat((10, 20, 30))
    q12d = q1 / q2
    q12m = q1 * q2.inv()
    assert np.all(q12d.q == q12m.q)
