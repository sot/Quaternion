import numpy as np
from Quaternion import Quat
from nose.tools import *

ra = 10.
dec = 20.
roll = 30.
q0 = Quat([ra,dec,roll])

def test_from_eq():
    q = Quat([ra, dec, roll])
    assert_almost_equal(q.q[0], 0.26853582)
    assert_almost_equal(q.q[1], -0.14487813)
    assert_almost_equal(q.q[2],  0.12767944)
    assert_almost_equal(q.q[3],  0.94371436)

def test_from_transform():
    """Initialize from inverse of q0 via transform matrix"""
    q = Quat(q0.transform.transpose())
    assert_almost_equal(q.q[0], -0.26853582)
    assert_almost_equal(q.q[1], 0.14487813)
    assert_almost_equal(q.q[2], -0.12767944)
    assert_almost_equal(q.q[3],  0.94371436)

def test_inv_eq():
    q = Quat(q0.equatorial)
    t = q.transform
    tinv = q.inv().transform
    t_tinv = np.dot(t, tinv)
    for v1, v2 in zip(t_tinv.flatten(), [1,0,0,0,1,0,0,0,1]):
        assert_almost_equal(v1, v2)

def test_inv_q():
    q = Quat(q0.q)
    t = q.transform
    tinv = q.inv().transform
    t_tinv = np.dot(t, tinv)
    for v1, v2 in zip(t_tinv.flatten(), [1,0,0,0,1,0,0,0,1]):
        assert_almost_equal(v1, v2)

