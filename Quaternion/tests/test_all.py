# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pytest

from Quaternion import Quat, QuatDescriptor, normalize, quat_mult, quat_to_equatorial


def indices(t):
    import itertools

    for k in itertools.product(*[range(i) for i in t]):
        yield k


def normalize_angles(x, xmin, xmax):
    while np.any(x >= xmax):
        x -= np.where(x > xmax, 360, 0)
    while np.any(x < xmin):
        x += np.where(x < xmin, 360, 0)


ra = 10.0
dec = 20.0
roll = 30.0
q0 = Quat([ra, dec, roll])


equatorial_23 = np.array(
    [
        [[10, 20, 30], [10, 20, -30], [10, -60, 30]],
        [[10, 20, 0], [10, 50, 30], [10, -50, -30]],
    ],
    dtype=float,
)

q_23 = np.zeros(equatorial_23[..., 0].shape + (4,))
for _i, _j in indices(equatorial_23.shape[:-1]):
    q_23[_i, _j] = Quat(equatorial_23[_i, _j]).q

transform_23 = np.zeros(equatorial_23[..., 0].shape + (3, 3))
for _i, _j in indices(transform_23.shape[:-2]):
    transform_23[_i, _j] = Quat(equatorial_23[_i, _j]).transform


def test_shape():
    q = Quat(
        q=np.zeros(
            4,
        )
    )
    assert q.shape == ()
    with pytest.raises(AttributeError):
        q.shape = (4,)


def test_init_exceptions():
    with pytest.raises(TypeError):
        _ = Quat(q=np.zeros((3,)))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(equatorial=np.zeros((4,)))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(transform=np.zeros((4,)))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(np.zeros((2,)))  # old-style API, wrong shape
    with pytest.raises(TypeError):
        _ = Quat(np.zeros((5,)))  # old-style API, wrong shape
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
        _ = Quat(
            equatorial=np.zeros(3), transform=np.zeros((3, 3))
        )  # too many arguments
    with pytest.raises(ValueError):
        # too many arguments
        _ = Quat(q=np.zeros(4), transform=np.zeros((3, 3)), equatorial=np.zeros(3))
    with pytest.raises(ValueError):
        _ = Quat(q=[[[1.0, 0.0, 0.0, 1.0]]])  # q not normalized
    with pytest.raises(ValueError):
        _ = Quat(q=[[[0.1, 0.0, 0.0, 0.1]]])  # q not normalized
    with pytest.raises(ValueError):
        _ = Quat([0, 1, "s"])  # could not convert string to float


def test_from_q():
    q = [0.26853582, -0.14487813, 0.12767944, 0.94371436]
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
    assert np.allclose(q.q[2], 0.12767944)
    assert np.allclose(q.q[3], 0.94371436)
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
    q = Quat(equatorial=[10.0, 20.0, 30.0])
    assert np.array(q.ra0).shape == ()
    assert np.array(q.roll0).shape == ()
    assert np.array(q.ra).shape == ()
    assert np.array(q.dec).shape == ()
    assert np.array(q.roll).shape == ()
    assert np.array(q.yaw).shape == ()
    assert np.array(q.pitch).shape == ()
    assert q.q.shape == (4,)
    assert q.equatorial.shape == (3,)
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
    assert np.allclose(q.q[3], 0.94371436)

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

    t = [
        [
            [
                [9.25416578e-01, -3.18795778e-01, -2.04874129e-01],
                [1.63175911e-01, 8.23172945e-01, -5.43838142e-01],
                [3.42020143e-01, 4.69846310e-01, 8.13797681e-01],
            ]
        ]
    ]
    q = Quat(transform=t)
    assert q.q.shape == (1, 1, 4)

    q = Quat(transform=transform_23)
    assert np.allclose(q.q, q_23)
    # to compare roll, it has to be normalized to within a fixed angular range (0, 360).
    eq = np.array(q.equatorial)
    normalize_angles(eq[..., -1], 0, 360)
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
    flip = np.sign(q_23[..., -1]).reshape((2, 3, 1))
    assert np.allclose(q.q, q_23 * flip)
    # to compare roll, it has to be normalized to within a fixed angular range (0, 360).
    eq = np.array(q.equatorial)
    normalize_angles(eq[..., -1], 0, 360)
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
    assert repr(q) == "<Quat q1=0.02632421 q2=-0.01721736 q3=0.00917905 q4=0.99946303>"

    class SubQuat(Quat):
        pass

    q = SubQuat([1, 2, 3])
    assert (
        repr(q) == "<SubQuat q1=0.02632421 q2=-0.01721736 q3=0.00917905 q4=0.99946303>"
    )

    q = Quat(equatorial=[[1, 2, 3]])
    assert (
        repr(q) == "Quat(array([[ 0.02632421, -0.01721736,  0.00917905,  0.99946303]]))"
    )


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
    assert (q1 * q3).shape != q1.shape
    assert (q1 * q3).shape == q3.shape


def test_mult_vectorized():
    q1 = Quat(q=q_23[:1, :2])  # (shape (2,1)
    q2 = Quat(q=q_23[1:, 1:])  # (shape (2,1)
    assert q1.q.shape == q2.q.shape
    q12 = q1 * q2
    assert q12.q.shape == q1.q.shape


def test_normalize():
    a = [[[1.0, 0.0, 0.0, 1.0]]]
    b = normalize(a)
    assert np.array(a).shape == b.shape
    assert np.isclose(np.sum(b**2), 1.0, rtol=0, atol=1e-12)

    # Check special case for an exact zero input
    a2 = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    with pytest.warns(UserWarning, match="Normalizing quaternion with zero norm"):
        b2 = normalize(a2)
    assert np.array(a2).shape == b2.shape
    assert np.allclose(np.sum(b2**2, axis=-1), 1.0, rtol=0, atol=1e-12)
    assert np.all(b2[1] == [0.0, 0.0, 0.0, 1.0])


def test_copy():
    # data members must be copies so they are not modified by accident
    q = np.array(q_23[0, 0])
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
    print(f"ra={q.ra:.5f}, dec={q.dec:.5f}, roll={q.roll:.5f}")


def test_scalar_attribute_types():
    q = Quat(equatorial=[10, 20, 30])
    attrs = ["ra", "dec", "roll", "ra0", "roll0", "pitch", "yaw", "transform", "q"]
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


def test_mult_and_dq_broadcasted():
    """Test mult and delta quat of Quats with different but broadcastable shapes."""
    q2 = Quat(equatorial=np.arange(18).reshape(3, 2, 3))
    q1 = Quat(equatorial=[[10, 20, 30], [40, 50, 60]])
    q0 = Quat(equatorial=[10, 20, 30])
    # (3,2) * () = (3,2)
    q20 = q2 * q0
    dq20 = q2.dq(q0)
    assert q20.shape == (3, 2)
    assert dq20.shape == (3, 2)
    for ii in range(3):
        for jj in range(2):
            qq = q2[ii, jj] * q0
            dq = q2[ii, jj].dq(q0)
            assert np.allclose(qq.q, q20.q[ii, jj])
            assert np.allclose(dq.q, dq20.q[ii, jj])

    # (3,2) * (2,) = (3,2)
    q21 = q2 * q1
    dq21 = q2.dq(q1)
    assert q21.shape == (3, 2)
    assert dq21.shape == (3, 2)
    for ii in range(3):
        for jj in range(2):
            qq = q2[ii, jj] * q1[jj]
            dq = q2[ii, jj].dq(q1[jj])
            assert np.allclose(qq.q, q21.q[ii, jj])
            assert np.allclose(dq.q, dq21.q[ii, jj])


def test_array_attribute_types():
    q = Quat(equatorial=[[10, 20, 30]])  # 1-d
    attrs = ["ra", "dec", "roll", "ra0", "roll0", "pitch", "yaw", "transform", "q"]
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
    filename = os.path.join(os.path.dirname(__file__), "data", "quaternion-v3.4.1.pkl")
    with open(filename, "rb") as f:
        quaternions = pickle.load(f)
    for q in quaternions:
        assert np.all(
            np.isclose(q.q, [0.26853582, -0.14487813, 0.12767944, 0.94371436])
        )
        assert np.all(np.isclose(q.equatorial, [10.0, 20.0, 30.0]))
        assert np.all(
            np.isclose(
                q.transform,
                [
                    [0.92541658, -0.31879578, -0.20487413],
                    [0.16317591, 0.82317294, -0.54383814],
                    [0.34202014, 0.46984631, 0.81379768],
                ],
            )
        )


def test_init_quat_from_attitude():
    # Basic tests for Quat.from_attitude
    q = Quat.from_attitude([Quat([0, 1, 2]), Quat([3, 4, 5])])
    # 1-d list of Quat
    assert np.allclose(q.equatorial, [[0, 1, 2], [3, 4, 5]])

    # From existing Quat
    q2 = Quat.from_attitude(q)
    assert np.all(q.q == q2.q)
    assert q is not q2

    # Normal Quat initializer: 3-element list implies equatorial
    q = Quat.from_attitude([10, 20, 30])
    assert np.allclose(q.equatorial, [10, 20, 30])

    # 2-d list of Quat
    q = Quat.from_attitude([[Quat([0, 1, 2]), Quat([3, 4, 5])]])
    assert np.allclose(q.equatorial, [[[0, 1, 2], [3, 4, 5]]])

    # 1-d list of equatorial floats
    q = Quat.from_attitude([[0, 1, 2], [3, 4, 5]])
    assert np.allclose(q.equatorial, [[[0, 1, 2], [3, 4, 5]]])

    # Heterogenous list of floats
    q = Quat.from_attitude([[0, 1, 2], [0, 1, 0, 0]])
    assert np.allclose(q.equatorial, [[0, 1, 2], [180, 0, 180]])

    # Bad 1-d list of equatorial floats
    with pytest.raises(ValueError, match="Float input must be a Nx3 or Nx4 array"):
        q = Quat.from_attitude([[0, 1, 2, 4, 5], [3, 4, 5, 6, 7]])

    # 1-d list of 4-vectors
    q_list = [[0, 0, 1, 0], [0, 1, 0, 0]]
    q = Quat.from_attitude(q_list)
    assert np.allclose(q.q, q_list)

    # Bad input
    with pytest.raises(ValueError, match="Unable to initialize Quat from 'blah'"):
        Quat.from_attitude("blah")


def test_rotate_x_to_vec_regress():
    """Note that truth values are just results from original code in Ska.quatutil.
    They have not been independently validated"""
    vec = [1, 2, 3]
    q = Quat.rotate_x_to_vec(vec)  # method='radec', default
    assert np.allclose(q.q, [0.2358142, -0.38155539, 0.4698775, 0.76027777])

    q = Quat.rotate_x_to_vec(vec, method="shortest")
    assert np.allclose(q.q, [0.0, -0.50362718, 0.33575146, 0.79600918])

    q = Quat.rotate_x_to_vec(vec, method="keep_z")
    assert np.allclose(q.q, [-0.16269544, -0.56161937, 0.22572786, 0.77920525])


@pytest.mark.parametrize("method", ("keep_z", "shortest", "radec"))
def test_rotate_x_to_vec_functional(method):
    vecs = np.random.random((100, 3)) - 0.5
    for vec in vecs:
        vec = vec / np.sqrt(np.sum(vec**2))  # noqa: PLW2901
        q = Quat.rotate_x_to_vec(vec, method)
        vec1 = np.dot(q.transform, [1.0, 0, 0])
        assert np.allclose(vec, vec1)

        if method == "radec":
            assert np.isclose(q.roll, 0.0)
        elif method == "keep_z":
            vec1 = np.dot(q.transform, [0, 0, 1.0])
            assert np.isclose(vec1[1], 0.0)


def test_rotate_x_to_vec_bad_method():
    with pytest.raises(ValueError, match="method must be one of"):
        Quat.rotate_x_to_vec([1, 2, 3], "not-a-method")


def test_rotate_about_vec():
    q = Quat([10, 20, 30])
    q2 = q.rotate_about_vec([0, 0, 10], 25)
    assert np.allclose(q2.equatorial, [10 + 25, 20, 30])

    q2 = q.rotate_about_vec([-10, 0, 0], 180)
    assert np.allclose(q2.equatorial, [350.0, -20.0, 210.0])


def test_rotate_about_vec_exceptions():
    q1 = Quat([10, 20, 30])
    q2 = Quat(equatorial=[[10, 20, 30], [1, 2, 3]])
    with pytest.raises(ValueError, match="vec must be a single 3-vector"):
        q1.rotate_about_vec([[1, 2, 3], [4, 5, 6]], 25)

    with pytest.raises(ValueError, match="alpha must be a scalar"):
        q1.rotate_about_vec([1, 2, 3], [25, 50])

    with pytest.raises(ValueError, match="quaternion must be a scalar"):
        q2.rotate_about_vec([1, 2, 3], 25)


@pytest.mark.parametrize("attr", ["q", "equatorial", "transform"])
def test_setting_different_shape(attr):
    q0 = Quat([1, 2, 3])
    q1 = Quat(equatorial=[[3, 1, 2], [4, 5, 6]])
    assert q1.shape == (2,)
    val = getattr(q1, attr)
    setattr(q0, attr, val)
    assert q0.shape == q1.shape
    assert np.all(getattr(q0, attr) == getattr(q1, attr))


def test_quat_to_equatorial():
    ras = np.arange(0, 361, 30)
    decs = np.arange(-90, 91, 30)
    rolls = np.arange(0, 361, 30)
    for ra in ras:
        for dec in decs:
            for roll in rolls:
                q = Quat([ra, dec, roll])
                eq0 = Quat(q=q.q).equatorial
                eq1 = quat_to_equatorial(q.q)
                assert np.allclose(eq0, eq1, rtol=0, atol=1e-10)


def test_quat_mult():
    ras = np.arange(0, 361, 30)
    decs = np.arange(-90, 91, 30)
    rolls = np.arange(0, 361, 30)
    for ra0, ra1 in zip(ras[:-1], ras[1:]):
        for dec0, dec1 in zip(decs[:-1], decs[1:]):
            for roll0, roll1 in zip(rolls[:-1], rolls[1:]):
                q0 = Quat([ra0, dec0, roll0])
                q1 = Quat([ra1, dec1, roll1])
                q01_0 = (q0 * q1).q
                q01_1 = quat_mult(q0.q, q1.q)
                assert np.allclose(q01_0, q01_1, rtol=0, atol=1e-10)


def test_quat_descriptor_not_required_no_default():
    @dataclass
    class MyClass:
        quat: Quat | None = QuatDescriptor()

    obj = MyClass()
    assert obj.quat is None

    obj = MyClass(quat=[10, 20, 30])
    assert isinstance(obj.quat, Quat)
    assert np.allclose(obj.quat.equatorial, [10, 20, 30], rtol=0, atol=1e-10)
    assert np.allclose(
        obj.quat.q,
        [0.26853582, -0.14487813, 0.12767944, 0.94371436],
        rtol=0,
        atol=1e-8,
    )


def test_quat_descriptor_is_required():
    @dataclass
    class MyClass:
        quat: Quat = QuatDescriptor(required=True)

    obj = MyClass([10, 20, 30])
    assert np.allclose(obj.quat.equatorial, [10, 20, 30], rtol=0, atol=1e-10)

    with pytest.raises(
        ValueError, match="attribute 'quat' is required and cannot be set to None"
    ):
        MyClass()


def test_quat_descriptor_has_default():
    @dataclass
    class MyClass:
        quat: Quat = QuatDescriptor(default=(10, 20, 30))

    obj = MyClass()
    assert np.allclose(obj.quat.equatorial, [10, 20, 30], rtol=0, atol=1e-10)

    obj = MyClass(quat=[30, 40, 50])
    assert np.allclose(obj.quat.equatorial, [30, 40, 50], rtol=0, atol=1e-10)


def test_quat_descriptor_is_required_has_default_exception():
    with pytest.raises(
        ValueError, match="cannot set both 'required' and 'default' arguments"
    ):

        @dataclass
        class MyClass1:
            quat: Quat = QuatDescriptor(default=[10, 20, 30], required=True)
