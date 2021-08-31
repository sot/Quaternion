# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest

from .. import Quat


@pytest.mark.parametrize('attr', ['q', 'equatorial', 'transform'])
def test_getitem(attr):
    """Test getitem and iteration"""
    eqs = np.arange(24).reshape(2, 4, 3)
    qs = Quat(equatorial=eqs)
    for i in range(eqs.shape[0]):
        for j in range(eqs.shape[1]):
            val1 = getattr(qs[i, j], attr)
            val2 = getattr(qs, attr)[i, j]
            assert np.allclose(val1, val2)
            assert val1.shape == val2.shape

    for i in range(eqs.shape[0]):
        val1 = getattr(qs[i], attr)
        val2 = getattr(qs, attr)[i]
        assert np.allclose(val1, val2)
        assert val1.shape == val2.shape

    for i, q_i in enumerate(qs):
        val1 = getattr(q_i, attr)
        val2 = getattr(qs, attr)[i]
        assert np.allclose(val1, val2)
        assert val1.shape == val2.shape


shape_methods = ('copy', 'reshape', 'transpose', 'flatten', 'ravel', 'swapaxes',
                 'diagonal', 'squeeze', 'take')
@pytest.mark.parametrize('method', shape_methods)  # noqa
def test_shape_changing_methods(method):
    eqs = np.arange(4 * 1 * 3).reshape(4, 1, 3)
    q1 = Quat(equatorial=eqs)
    if method == 'take':
        args = (3, 0)
    elif method == 'squeeze':
        args = (-1,)
    elif method == 'reshape':
        args = (2, 2)
    elif method == 'swapaxes':
        args = (0, 1)
    else:
        args = ()

    val1 = getattr(q1, method)(*args).equatorial
    val20 = getattr(q1.equatorial[..., 0], method)(*args)
    val21 = getattr(q1.equatorial[..., 1], method)(*args)
    val22 = getattr(q1.equatorial[..., 2], method)(*args)
    val2 = np.stack([val20, val21, val22], axis=-1)
    assert val1.shape == val2.shape
    assert np.allclose(val1, val2)

    val1 = getattr(q1, method)(*args).q
    val20 = getattr(q1.q[..., 0], method)(*args)
    val21 = getattr(q1.q[..., 1], method)(*args)
    val22 = getattr(q1.q[..., 2], method)(*args)
    val23 = getattr(q1.q[..., 3], method)(*args)
    val2 = np.stack([val20, val21, val22, val23], axis=-1)
    assert val1.shape == val2.shape
    assert np.allclose(val1, val2)


def test_shape_changing_T_property():
    eqs = np.arange(4 * 1 * 3).reshape(4, 1, 3)
    q1 = Quat(equatorial=eqs)

    val1 = q1.T.equatorial
    val20 = q1.equatorial[..., 0].T
    val21 = q1.equatorial[..., 1].T
    val22 = q1.equatorial[..., 2].T
    val2 = np.stack([val20, val21, val22], axis=-1)

    assert val1.shape == val2.shape
    assert np.allclose(val1, val2)


applicable_methods = [
    (np.moveaxis, (0, 1)),
    (np.rollaxis, (1,)),
    (np.atleast_1d, ()),
    (np.atleast_2d, ()),
    (np.atleast_3d, ()),
    (np.expand_dims, (1,)),
    (np.broadcast_to, ((3, 4, 1),)),
    (np.flip, ()),
    (np.fliplr, ()),
    (np.flipud, ()),
    (np.rot90, ()),
    (np.roll, (1,)),
    (np.delete, ([0, 2], 0))
]


@pytest.mark.parametrize('method, args', applicable_methods)
def test_applicable_methods(method, args):
    eqs = np.arange(4 * 1 * 3).reshape(4, 1, 3)
    q1 = Quat(equatorial=eqs)

    val1 = method(q1, *args).equatorial
    val20 = method(q1.equatorial[..., 0], *args)
    val21 = method(q1.equatorial[..., 1], *args)
    val22 = method(q1.equatorial[..., 2], *args)
    val2 = np.stack([val20, val21, val22], axis=-1)

    assert val1.shape == val2.shape
    assert np.allclose(val1, val2)
