import numpy as np
from .. import Quat

ra = 10.
dec = 20.
roll = 30.
q0 = Quat([ra, dec, roll])


def test_from_eq():
    q = Quat([ra, dec, roll])
    print 'q.equatorial=', q.equatorial
    assert np.allclose(q.q[0], 0.26853582)
    assert np.allclose(q.q[1], -0.14487813)
    assert np.allclose(q.q[2],  0.12767944)
    assert np.allclose(q.q[3],  0.94371436)
    assert np.allclose(q.roll0, 30)
    assert np.allclose(q.ra0, 10)


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


def test_issue_1():
    # Test for numeric issue https://github.com/sot/Quaternion/issues/1
    quat = Quat((0, 0, 0))
    angle = 0
    while angle < 360:
        q = Quat((0, angle, 0))
        quat = q * quat
        quat.equatorial
        angle += 0.1
