# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
import pickle
import os

from .. import Quat, normalize


def indices(t):
    import itertools
    for k in itertools.product(*[range(i) for i in t]):
        yield k

def normalize_angles(x, xmin, xmax):
    while np.any(x >= xmax):
        x -= np.where(x > xmax, 360, 0)
    while np.any(x < xmin):
        x += np.where(x < xmin, 360, 0)


ra = 10.
dec = 20.
roll = 30.
q0 = Quat([ra, dec, roll])


equatorial_23 = np.array([[[10, 20, 30],
                           [10, 20, -30],
                           [10, -60, 30]],
                          [[10, 20, 0],
                           [10, 50, 30],
                           [10, -50, -30]]], dtype=float)

q_23 = np.zeros(equatorial_23[..., 0].shape+(4,))
for _i, _j in indices(equatorial_23.shape[:-1]):
    q_23[_i, _j] = Quat(equatorial_23[_i, _j]).q

transform_23 = np.zeros(equatorial_23[..., 0].shape+(3, 3))
for _i, _j in indices(transform_23.shape[:-2]):
    transform_23[_i, _j] = Quat(equatorial_23[_i, _j]).transform


def test_shape():
    q = Quat(q=np.zeros(4,))
    assert q.shape == ()
    with pytest.raises(AttributeError):
        q.shape = (4,)


def test_init_exceptions():
    with pytest.raises(TypeError):
        _ = Quat(q=np.zeros((3, )))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(equatorial=np.zeros((4, )))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(transform=np.zeros((4, )))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(np.zeros((2, )))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(np.zeros((5, )))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(equatorial_23)  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(q_23)  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(transform_23)  # old-style API, wrong shape
    with pytest.raises(ValueError):
        _ = Quat(q=np.zeros(4), transform=np.zeros((3, 3)))  # too many arguments
    with pytest.raises(ValueError):
        _ = Quat(q=np.zeros(4), equatorial=np.zeros(3))  # too many arguments
    with pytest.raises(ValueError):
        _ = Quat(equatorial=np.zeros(3), transform=np.zeros((3, 3)))  # too many arguments
    with pytest.raises(ValueError):
        # too many arguments
        _ = Quat(q=np.zeros(4), transform=np.zeros((3, 3)), equatorial=np.zeros(3))
    with pytest.raises(ValueError):
        _ = Quat(q=[[[1., 0., 0., 1.]]])  # q not normalized
    with pytest.raises(ValueError):
        _ = Quat([0,1,'s'])  # could not convert string to float


def test_from_q():
    q = [0.26853582, -0.14487813,  0.12767944,  0.94371436]
    q1 = Quat(q)
    q2 = Quat(q=q)
    q3 = Quat(q1)
    q = np.array(q)
    assert np.all(q1.q == q)
    assert np.all(q2.q == q)
    assert np.all(q3.q == q)


def test_from_eq():
    q = Quat([ra, dec, roll])
    assert np.allclose(q.q[0], 0.26853582)
    assert np.allclose(q.q[1], -0.14487813)
    assert np.allclose(q.q[2],  0.12767944)
    assert np.allclose(q.q[3],  0.94371436)
    assert np.allclose(q.roll0, 30)
    assert np.allclose(q.ra0, 10)
    assert q.pitch == -q.dec
    assert q.yaw == q.ra0

    q1 = Quat(equatorial=[ra, dec, roll])
    assert np.all(q1.q == q.q)


def test_from_eq_vectorized():
    # the following line would give unexpected results
    # because  the input is interpreted as a (non-vectorized) transform
    # the shape of the input is (3,3)
    # q = Quat(equatorial_23[0])

    # this is the proper way:
    q = Quat(equatorial=equatorial_23[0])
    assert q.q.shape == (3, 4)
    for i in indices(q.shape):
        # check that Quat(equatorial).q[i] == Quat(equatorial[i]).q
        assert np.all(q.q[i] == Quat(equatorial_23[0][i]).q)

    q = Quat(equatorial=equatorial_23)
    assert q.q.shape == (2, 3, 4)
    for i in indices(q.shape):
        # check that Quat(equatorial).q[i] == Quat(equatorial[i]).q
        assert np.all(q.q[i] == Quat(equatorial_23[i]).q)

    # test init from list
    q = Quat(equatorial=[ra, dec, roll])
    assert np.all(q.q == q0.q)

    q = Quat(equatorial=equatorial_23)
    assert np.all(q.q == q_23)
    assert np.all(q.equatorial == equatorial_23)
    assert np.all(q.transform == transform_23)

def test_from_eq_shapes():
    q = Quat(equatorial=[ 10., 20., 30.])
    assert np.array(q.ra0).shape == ()
    assert np.array(q.roll0).shape == ()
    assert np.array(q.ra).shape == ()
    assert np.array(q.dec).shape == ()
    assert np.array(q.roll).shape == ()
    assert np.array(q.yaw).shape == ()
    assert np.array(q.pitch).shape == ()
    assert q.q.shape == (4, )
    assert q.equatorial.shape == (3, )
    assert q.transform.shape == (3, 3)

    q = Quat(equatorial=equatorial_23[:1, :1])
    assert q.ra0.shape == (1, 1)
    assert q.roll0.shape == (1, 1)
    assert q.ra.shape == (1, 1)
    assert q.dec.shape == (1, 1)
    assert q.roll.shape == (1, 1)
    assert q.yaw.shape == (1, 1)
    assert q.pitch.shape == (1, 1)
    assert q.q.shape == (1, 1, 4)
    assert q.equatorial.shape == (1, 1, 3)
    assert q.transform.shape == (1, 1, 3, 3)


def test_transform_from_eq():
    q = Quat(equatorial=equatorial_23)
    assert q.transform.shape == (2, 3, 3, 3)
    for i in indices(q.shape):
        # check that
        # Quat(equatorial).transform[i] == Quat(equatorial[i]).transform
        assert np.all(q.transform[i] == Quat(equatorial_23[i]).transform)


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

    q1 = Quat(transform=q0.transform)
    assert np.all(q1.q == q.q)


def test_from_transform_vectorized():
    q = Quat(transform=transform_23)
    assert q.q.shape == (2, 3, 4)
    for i in indices(q.shape):
        # check that Quat(transform).q[i] == Quat(transform[i]).q
        assert np.all(q.q[i] == Quat(transform=transform_23[i]).q)

    q = Quat(transform=transform_23[:1, :1])
    assert q.q.shape == (1, 1, 4)

    t = [[[[9.25416578e-01, -3.18795778e-01, -2.04874129e-01],
           [1.63175911e-01, 8.23172945e-01, -5.43838142e-01],
           [3.42020143e-01, 4.69846310e-01, 8.13797681e-01]]]]
    q = Quat(transform=t)
    assert q.q.shape == (1, 1, 4)

    q = Quat(transform=transform_23)
    assert np.allclose(q.q, q_23)
    # to compare roll, it has to be normalized to within a fixed angular range (0, 360).
    eq = np.array(q.equatorial)
    normalize_angles(eq[...,-1], 0, 360)
    eq_23 = np.array(equatorial_23)
    normalize_angles(eq_23[..., -1], 0, 360)
    assert np.allclose(eq, eq_23)
    assert np.allclose(q.transform, transform_23)

def test_eq_from_transform():
    # this raises 'Unexpected negative norm' exception due to roundoff in copy/paste above
    # q = Quat(transform=transform_23)
    # assert q.equatorial.shape == (2, 3, 3)
    # assert np.allclose(q.equatorial, equatorial_23)

    t = np.zeros((4, 5, 3, 3))
    t[:] = q0.transform[np.newaxis][np.newaxis]
    q = Quat(transform=t)
    assert np.allclose(q.roll0, 30)
    assert np.allclose(q.ra0, 10)

    assert q.equatorial.shape == (4, 5, 3)


def test_from_q_vectorized():
    q = Quat(q=q_23)
    assert q.shape == (2, 3)
    # this also tests that quaternions with negative scalar component are flipped
    flip = np.sign(q_23[...,-1]).reshape((2,3,1))
    assert np.allclose(q.q, q_23*flip)
    # to compare roll, it has to be normalized to within a fixed angular range (0, 360).
    eq = np.array(q.equatorial)
    normalize_angles(eq[...,-1], 0, 360)
    eq_23 = np.array(equatorial_23)
    normalize_angles(eq_23[..., -1], 0, 360)
    assert np.allclose(eq, eq_23, rtol=0)
    assert np.allclose(q.transform, transform_23, rtol=0)

    q = Quat(q=q_23[0])
    assert q.shape == (3,)

    q = Quat(q=q_23[:1, :1])
    assert q.shape == (1, 1)


def test_inv_eq():
    q = Quat(q0.equatorial)
    t = q.transform
    tinv = q.inv().transform
    t_tinv = np.dot(t, tinv)
    for v1, v2 in zip(t_tinv.flatten(), [1, 0, 0, 0, 1, 0, 0, 0, 1]):
        assert np.allclose(v1, v2)


def test_inv_q():
    q = Quat(q0.q)
    assert q.q.shape == q.inv().q.shape
    t = q.transform
    tinv = q.inv().transform
    t_tinv = np.dot(t, tinv)
    for v1, v2 in zip(t_tinv.flatten(), [1, 0, 0, 0, 1, 0, 0, 0, 1]):
        assert np.allclose(v1, v2)


def test_inv_vectorized():
    q1 = Quat(q=q_23[:1, :1])
    assert q1.q.shape == (1, 1, 4)
    q1_inv = q1.inv()
    assert q1_inv.q.shape == q1.q.shape
    for i in indices(q1.shape):
        # check that Quat(q).inv().q[i] == Quat(q[i]).inv().q
        assert np.all(q1_inv.q[i] == Quat(q=q1.q[i]).inv().q)

def test_dq():
    q1 = Quat((20, 30, 0))
    q2 = Quat((20, 30.1, 1))
    dq = q1.dq(q2)
    assert np.allclose(dq.equatorial, (0, 0.1, 1))

    # same from array instead of Quat
    dq = q1.dq(q2.q)
    assert np.allclose(dq.equatorial, (0, 0.1, 1))


def test_dq_vectorized():
    q1 = Quat(q=q_23[:1, :2])
    q2 = Quat(q=q_23[1:, 1:])
    assert q1.q.shape == q2.q.shape

    dq = q1.dq(q2)
    assert dq.q.shape == q1.q.shape  # shape (1,2,4)

    # same but with array argument instead of Quat
    dq2 = q1.dq(q=q2.q)
    assert dq2.q.shape == dq.q.shape
    assert np.all(dq2.q == dq.q)

    for i in indices(q1.shape):
        # check that Quat(q1).dq(q2).q[i] == Quat(q1[i]).dq(q2[i]).q
        assert np.all(dq.q[i] == Quat(q=q1.q[i]).dq(Quat(q=q2.q[i])).q)

    # note that both quaternions have same _internal_ shape, should this fail?
    q1 = Quat((20, 30, 0))
    q2 = Quat(equatorial=[[20, 30.1, 1]])
    assert np.allclose(q1.dq(q2).equatorial, [[0, 0.1, 1]])
    assert np.allclose(q1.dq(q=q2.q).equatorial, [[0, 0.1, 1]])
    assert np.allclose(q1.dq(equatorial=q2.equatorial).equatorial, [[0, 0.1, 1]])
    assert np.allclose(q1.dq(transform=q2.transform).equatorial, [[0, 0.1, 1]])
    # and the interface is the same as the constructor:
    with pytest.raises(TypeError):
        q1.dq(q2.q)
    with pytest.raises(TypeError):
        q1.dq(q2.equatorial)
    with pytest.raises(TypeError):
        q1.dq(q2.transform)


def test_vector_to_scalar_correspondence():
    """
    Simple test that all possible transform pathways give the same
    answer when done in vectorized form as they do for the scalar version.
    """
    atol = 1e-12

    # Input equatorial has roll not in 0:360, so fix that for comparisons.
    eq_23 = equatorial_23.copy()
    normalize_angles(eq_23[..., -1], 0, 360)

    # Compare vectorized computations for all possible input/output combos
    # with the same for the scalar calculation.
    q = Quat(equatorial=equatorial_23)
    assert np.all(q.q == q_23)
    assert np.all(q.equatorial == equatorial_23)
    assert np.all(q.transform == transform_23)

    q = Quat(q=q_23)
    assert np.all(q.q == q_23)
    assert np.allclose(q.equatorial, eq_23, rtol=0, atol=atol)
    assert np.allclose(q.transform, transform_23, rtol=0, atol=atol)

    q = Quat(transform=transform_23)
    assert np.allclose(q.q, q_23, rtol=0, atol=atol)
    assert np.allclose(q.equatorial, eq_23, rtol=0, atol=atol)
    assert np.all(q.transform == transform_23)


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

    q = Quat(equatorial=[[1, 2, 3]])
    assert repr(q) == 'Quat(array([[ 0.02632421, -0.01721736,  0.00917905,  0.99946303]]))'


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
        _ = quat.equatorial
        angle += 0.1


def test_div_mult():
    q1 = Quat((1, 2, 3))
    q2 = Quat((10, 20, 30))
    q12d = q1 / q2
    assert q1.shape == q12d.shape
    assert q1.shape == q1.inv().shape
    q12m = q1 * q2.inv()
    assert q1.shape == q12m.shape
    assert np.all(q12d.q == q12m.q)

    q3 = Quat(equatorial=[[10, 20, 30]])
    assert (q1*q3).shape != q1.shape
    assert (q1*q3).shape == q3.shape

def test_mult_vectorized():
    q1 = Quat(q=q_23[:1, :2])  # (shape (2,1)
    q2 = Quat(q=q_23[1:, 1:])  # (shape (2,1)
    assert q1.q.shape == q2.q.shape
    q12 = q1*q2
    assert q12.q.shape == q1.q.shape


def test_normalize():
    a = [[[1., 0., 0., 1.]]]
    b = normalize(a)
    assert np.isclose(np.sum(b**2), 1)


def test_copy():
    # data members must be copies so they are not modified by accident
    q = np.array(q_23[0,0])
    q1 = Quat(q=q)
    q[-1] = 0
    assert q1.q[-1] != 0

    # this one passes
    t = np.array(transform_23)
    q1 = Quat(transform=t)
    t[-1] = 0
    assert not np.all(q1.transform == t)

    # this one passes
    eq = np.array([10, 90, 30])
    q1 = Quat(equatorial=eq)
    eq[-1] = 0
    assert not np.all(q1.equatorial == eq)


def test_format():
    # this is to test standard usage downstream
    q = Quat(q_23[0, 0])
    print(f'ra={q.ra:.5f}, dec={q.dec:.5f}, roll={q.roll:.5f}')


def test_scalar_attribute_types():
    q = Quat(equatorial=[10, 20, 30])
    attrs = ['ra', 'dec', 'roll', 'ra0', 'roll0', 'pitch', 'yaw', 'transform', 'q']
    types = [np.float64] * 7 + [np.ndarray] * 2

    # All returned as scalars
    for attr, typ in zip(attrs, types):
        assert type(getattr(q, attr)) is typ

    q2 = Quat(transform=q.transform.astype(np.float32))
    for attr, typ in zip(attrs, types):
        assert type(getattr(q2, attr)) is typ

    q2 = Quat(q=q.q.astype(np.float32))
    for attr, typ in zip(attrs, types):
        assert type(getattr(q, attr)) is typ


def test_array_attribute_types():
    q = Quat(equatorial=[[10, 20, 30]])  # 1-d
    attrs = ['ra', 'dec', 'roll', 'ra0', 'roll0', 'pitch', 'yaw', 'transform', 'q']
    shapes = [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1, 3, 3), (1, 4)]

    # All returned as shape (1,) array
    for attr, shape in zip(attrs, shapes):
        assert type(getattr(q, attr)) is np.ndarray
        assert getattr(q, attr).shape == shape

    q2 = Quat(transform=q.transform.astype(np.float32))
    for attr, shape in zip(attrs, shapes):
        assert type(getattr(q2, attr)) is np.ndarray
        assert getattr(q, attr).shape == shape

    q2 = Quat(q=q.q.astype(np.float32))
    for attr, shape in zip(attrs, shapes):
        assert type(getattr(q, attr)) is np.ndarray
        assert getattr(q, attr).shape == shape


def test_pickle():
    """
    Pickle file generated using Quaternion v3.4.1:

        from Quaternion import Quat
        import pickle
        q = Quat([10., 20., 30.])
        quats = [Quat(q.q), Quat(q.transform), Quat(q.equatorial)]
        quats.append(q)
        with open('quaternion-v3.4.1.pkl', 'wb') as f:
            pickle.dump(quats, f)
    """
    # testing we can unpickle older versions
    filename = os.path.join(os.path.dirname(__file__), 'data', 'quaternion-v3.4.1.pkl')
    with open(filename, 'rb') as f:
        quaternions = pickle.load(f)
    for q in quaternions:
        assert np.all(np.isclose(q.q, [0.26853582, -0.14487813, 0.12767944, 0.94371436]))
        assert np.all(np.isclose(q.equatorial, [ 10.,  20.,  30.]))
        assert np.all(np.isclose(q.transform, [[ 0.92541658, -0.31879578, -0.20487413],
                                               [ 0.16317591,  0.82317294, -0.54383814],
                                               [ 0.34202014,  0.46984631,  0.81379768]]))
