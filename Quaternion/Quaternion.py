# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Quaternion provides a class for manipulating quaternion objects.  This class provides:

   - a convenient constructor to convert to/from Euler Angles (RA,Dec,Roll)
       to/from quaternions
   - class methods to multiply and divide quaternions

:Copyright: Smithsonian Astrophysical Observatory (2010)
:Author: Jean Connelly (jconnelly@cfa.harvard.edu)
"""
# Copyright (c) 2010, Smithsonian Astrophysical Observatory
# All rights reserved.
##
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither the name of the Smithsonian Astrophysical Observatory nor the
# names of its contributors may be used to endorse or promote products
# derived from this software without specific prior written permission.
##
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from math import cos, sin, radians, degrees, atan2, sqrt


class Quat(object):

    """
    Quaternion class

    Example usage::

     >>> from Quaternion import Quat
     >>> quat = Quat((12,45,45))
     >>> quat.ra, quat.dec, quat.roll
     (12, 45, 45)
     >>> quat.q
     array([ 0.38857298, -0.3146602 ,  0.23486498,  0.8335697 ])
     >>> q2 = Quat(quat.q)
     >>> q2.ra
     12.0

    Multiplication and division operators are overloaded for the class to
    perform appropriate quaternion multiplication and division.

    Quaternion composition as a multiplication q = q1 * q2 is equivalent to
    applying the q2 transform followed by the q1 transform.  Another way to
    express this is::

      q = Quat(numpy.dot(q1.transform, q2.transform))

    Example usage::

      >>> q1 = Quat((20, 30, 0))
      >>> q2 = Quat((0, 0, 40))
      >>> (q1 * q2).equatorial
      array([20., 30., 40.])

    This example first rolls about X by 40 degrees, then rotates that rolled frame
    to RA=20 and Dec=30.  Doing the composition in the other order does a roll about
    (the original) X-axis of the (RA, Dec) = (20, 30) frame, yielding a non-intuitive
    though correct result::

      >>> (q2 * q1).equatorial
      array([ 353.37684725,   34.98868888,   47.499696  ])

     Be aware that for Chandra operations the correct formalism for applying a
     delta quaternion ``dq`` to maneuver from ``q1`` to ``q2`` is::

       >>> q2 = q1 * dq
       >>> dq = q1.inv() * q2

     The quaternion returned by ``q1 / q2`` using the divide operator represents the
     delta quaternion in the INERTIAL FRAME, which is not what is used by the
     spacecraft to represent maneuvers.

     Instead use the ``dq`` method (or equivalently ``q1.inv() * q2``) for computing a
     delta quaternion to maneuver from ``q1`` to ``q2``::

       >>> dq = q1.dq(q2)

    :param attitude: initialization attitude for quat

    ``attitude`` may be:
      * another Quat
      * a 4 element array (expects x,y,z,w quat form)
      * a 3 element array (expects ra,dec,roll in degrees)
      * a 3x3 transform/rotation matrix

    """

    def __init__(self, attitude):
        self._q = None
        self._equatorial = None
        self._T = None
        # checks to see if we've been passed a Quat
        if isinstance(attitude, Quat):
            self._set_q(attitude.q)
        else:
            # make it an array and check to see if it is a supported shape
            attitude = np.array(attitude)
            if len(attitude) == 4:
                self._set_q(attitude)
            elif attitude.shape == (3, 3):
                self._set_transform(attitude)
            elif attitude.shape == (3,):
                self._set_equatorial(attitude)
            else:
                raise TypeError(
                    "attitude is not one of possible types (3 or 4 elements, Quat, or 3x3 matrix)")

    def _set_q(self, q):
        """
        Set the value of the 4 element quaternion vector

        :param q: list or array of normalized quaternion elements
        """
        q = np.array(q)
        if abs(np.sum(q**2) - 1.0) > 1e-6:
            raise ValueError(
                'Quaternion must be normalized so sum(q**2) == 1; use Quaternion.normalize')
        self._q = (q if q[3] > 0 else -q)
        # Erase internal values of other representations
        self._equatorial = None
        self._T = None

    def _get_q(self):
        """
        Retrieve 4-vector of quaternion elements in [x, y, z, w] form

        :rtype: numpy array

        """
        if self._q is None:
            # Figure out q from available values, doing nothing others are not defined
            if self._equatorial is not None:
                self._q = self._equatorial2quat()
            elif self._T is not None:
                self._q = self._transform2quat()
        return self._q

    # use property to make this get/set automatic
    q = property(_get_q, _set_q)

    def __repr__(self):
        q = self.q
        return ('<{} q1={:.8f} q2={:.8f} q3={:.8f} q4={:.8f}>'
                .format(self.__class__.__name__, q[0], q[1], q[2], q[3]))

    def _set_equatorial(self, equatorial):
        """Set the value of the 3 element equatorial coordinate list [RA,Dec,Roll]
           expects values in degrees
           bounds are not checked

           :param equatorial: list or array [ RA, Dec, Roll] in degrees

        """
        self._equatorial = np.array(equatorial)

    def _get_equatorial(self):
        """Retrieve [RA, Dec, Roll]

        :rtype: numpy array
        """
        if self._equatorial is None:
            if self._q is not None:
                self._equatorial = self._quat2equatorial()
            elif self._T is not None:
                self._q = self._transform2quat()
                self._equatorial = self._quat2equatorial()
        return self._equatorial

    equatorial = property(_get_equatorial, _set_equatorial)

    def _get_ra(self):
        """Retrieve RA term from equatorial system in degrees"""
        return self.equatorial[0]

    def _get_dec(self):
        """Retrieve Dec term from equatorial system in degrees"""
        return self.equatorial[1]

    def _get_roll(self):
        """Retrieve Roll term from equatorial system in degrees"""
        return self.equatorial[2]

    ra = property(_get_ra)
    dec = property(_get_dec)
    roll = property(_get_roll)

    def _get_zero(self, attr):
        """
        Return a version of attr that is between -180 <= val < 180
        """
        if not hasattr(self, '_' + attr):
            val = getattr(self, attr) % 360.0
            if val >= 180:
                val -= 360
        return val

    @property
    def ra0(self):
        """
        Return quaternion RA in the range -180 <= roll < 180.
        """
        if not hasattr(self, '_ra0'):
            self._ra0 = self._get_zero('ra')
        return self._ra0

    @property
    def roll0(self):
        """
        Return quaternion roll in the range -180 <= roll < 180.
        """
        if not hasattr(self, '_roll0'):
            self._roll0 = self._get_zero('roll')
        return self._roll0

    @property
    def pitch(self):
        """
        Return quaternion pitch (same as -dec)
        """
        return -self.dec

    @property
    def yaw(self):
        """
        Return quaternion yaw (same as ra0)
        """
        return self.ra0

    def _set_transform(self, T):
        """
        Set the value of the 3x3 rotation/transform matrix

        :param T: 3x3 array/numpy array
        """
        transform = np.array(T)
        self._T = transform

    def _get_transform(self):
        """
        Retrieve the value of the 3x3 rotation/transform matrix

        :returns: 3x3 rotation/transform matrix
        :rtype: numpy array

        """
        if self._T is None:
            if self._q is not None:
                self._T = self._quat2transform()
            elif self._equatorial is not None:
                self._T = self._equatorial2transform()
        return self._T

    transform = property(_get_transform, _set_transform)

    def _quat2equatorial(self):
        """
        Determine Right Ascension, Declination, and Roll for the object quaternion

        :returns: RA, Dec, Roll
        :rtype: numpy array [ra,dec,roll]
        """

        q = self.q
        q2 = self.q**2

        # calculate direction cosine matrix elements from $quaternions
        xa = q2[0] - q2[1] - q2[2] + q2[3]
        xb = 2 * (q[0] * q[1] + q[2] * q[3])
        xn = 2 * (q[0] * q[2] - q[1] * q[3])
        yn = 2 * (q[1] * q[2] + q[0] * q[3])
        zn = q2[3] + q2[2] - q2[0] - q2[1]

        # Due to numerical precision this can go negative.  Allow *slightly* negative
        # values but raise an exception otherwise.
        one_minus_xn2 = 1 - xn**2
        if one_minus_xn2 < 0:
            if one_minus_xn2 < -1e-12:
                raise ValueError('Unexpected negative norm: {}'.format(one_minus_xn2))
            one_minus_xn2 = 0

        # ; calculate RA, Dec, Roll from cosine matrix elements
        ra = degrees(atan2(xb, xa))
        dec = degrees(atan2(xn, sqrt(one_minus_xn2)))
        roll = degrees(atan2(yn, zn))
        if (ra < 0):
            ra += 360
        if (roll < 0):
            roll += 360

        return np.array([ra, dec, roll])

    def _quat2transform(self):
        """
        Transform a unit quaternion into its corresponding rotation matrix (to
        be applied on the right side).

        :returns: transform matrix
        :rtype: numpy array

        """
        x, y, z, w = self.q
        xx2 = 2 * x * x
        yy2 = 2 * y * y
        zz2 = 2 * z * z
        xy2 = 2 * x * y
        wz2 = 2 * w * z
        zx2 = 2 * z * x
        wy2 = 2 * w * y
        yz2 = 2 * y * z
        wx2 = 2 * w * x

        rmat = np.empty((3, 3), float)
        rmat[0, 0] = 1. - yy2 - zz2
        rmat[0, 1] = xy2 - wz2
        rmat[0, 2] = zx2 + wy2
        rmat[1, 0] = xy2 + wz2
        rmat[1, 1] = 1. - xx2 - zz2
        rmat[1, 2] = yz2 - wx2
        rmat[2, 0] = zx2 - wy2
        rmat[2, 1] = yz2 + wx2
        rmat[2, 2] = 1. - xx2 - yy2

        return rmat

    def _equatorial2quat(self):
        """Dummy method to return return quat.

        :returns: quaternion
        :rtype: Quat

        """
        return self._transform2quat()

    def _equatorial2transform(self):
        """Construct the transform/rotation matrix from RA,Dec,Roll

        :returns: transform matrix
        :rtype: 3x3 numpy array

        """
        ra = radians(self._get_ra())
        dec = radians(self._get_dec())
        roll = radians(self._get_roll())
        ca = cos(ra)
        sa = sin(ra)
        cd = cos(dec)
        sd = sin(dec)
        cr = cos(roll)
        sr = sin(roll)

        # This is the transpose of the transformation matrix (related to translation
        # of original perl code
        rmat = np.array([[ca * cd,                    sa * cd,                  sd],
                         [-ca * sd * sr - sa * cr,   -sa * sd * sr + ca * cr,   cd * sr],
                         [-ca * sd * cr + sa * sr,   -sa * sd * cr - ca * sr,   cd * cr]])

        return rmat.transpose()

    def _transform2quat(self):
        """Construct quaternion from the transform/rotation matrix

        :returns: quaternion formed from transform matrix
        :rtype: numpy array
        """

        # Code was copied from perl PDL code that uses backwards index ordering
        T = self.transform.transpose()
        den = np.array([1.0 + T[0, 0] - T[1, 1] - T[2, 2],
                        1.0 - T[0, 0] + T[1, 1] - T[2, 2],
                        1.0 - T[0, 0] - T[1, 1] + T[2, 2],
                        1.0 + T[0, 0] + T[1, 1] + T[2, 2]])

        max_idx = np.flatnonzero(den == max(den))[0]

        q = np.zeros(4)
        q[max_idx] = 0.5 * sqrt(max(den))
        denom = 4.0 * q[max_idx]
        if (max_idx == 0):
            q[1] = (T[1, 0] + T[0, 1]) / denom
            q[2] = (T[2, 0] + T[0, 2]) / denom
            q[3] = -(T[2, 1] - T[1, 2]) / denom
        if (max_idx == 1):
            q[0] = (T[1, 0] + T[0, 1]) / denom
            q[2] = (T[2, 1] + T[1, 2]) / denom
            q[3] = -(T[0, 2] - T[2, 0]) / denom
        if (max_idx == 2):
            q[0] = (T[2, 0] + T[0, 2]) / denom
            q[1] = (T[2, 1] + T[1, 2]) / denom
            q[3] = -(T[1, 0] - T[0, 1]) / denom
        if (max_idx == 3):
            q[0] = -(T[2, 1] - T[1, 2]) / denom
            q[1] = -(T[0, 2] - T[2, 0]) / denom
            q[2] = -(T[1, 0] - T[0, 1]) / denom

        return q

    def __div__(self, quat2):
        """
        Divide one quaternion by another.

        Example usage::

         >>> q1 = Quat((20,30,40))
         >>> q2 = Quat((30,40,50))
         >>> q = q1 / q2

        Performs the operation as q1 * inverse(q2) which is equivalent to
        the inverse(q2) transform followed by the q1 transform.  See the __mul__
        operator help for more explanation on composing quaternions.

        :returns: product q1 * inverse q2
        :rtype: Quat

        """
        return self * quat2.inv()

    __truediv__ = __div__

    def __mul__(self, quat2):
        """
        Multiply quaternion by another.

        Quaternion composition as a multiplication q = q1 * q2 is equivalent to
        applying the q2 transform followed by the q1 transform.  Another way to
        express this is::

          q = Quat(numpy.dot(q1.transform, q2.transform))

        Example usage::

          >>> q1 = Quat((20,30,0))
          >>> q2 = Quat((0,0,40))
          >>> (q1 * q2).equatorial
          array([20., 30., 40.])

        This example first rolls about X by 40 degrees, then rotates that rolled frame
        to RA=20 and Dec=30.  Doing the composition in the other order does a roll about
        (the original) X-axis of the (RA, Dec) = (20, 30) frame, yielding a non-intuitive
        though correct result::

          >>> (q2 * q1).equatorial
          array([ 353.37684725,   34.98868888,   47.499696  ])

        :returns: product q1 * q2
        :rtype: Quat

        """
        q1 = self.q
        q2 = quat2.q
        mult = np.zeros(4)
        mult[0] = q1[3] * q2[0] - q1[2] * q2[1] + q1[1] * q2[2] + q1[0] * q2[3]
        mult[1] = q1[2] * q2[0] + q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3]
        mult[2] = -q1[1] * q2[0] + q1[0] * q2[1] + q1[3] * q2[2] + q1[2] * q2[3]
        mult[3] = -q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] + q1[3] * q2[3]
        return Quat(mult)

    def inv(self):
        """
        Invert the quaternion

        :returns: inverted quaternion
        :rtype: Quat
        """
        return Quat([self.q[0], self.q[1], self.q[2], -self.q[3]])

    def dq(self, q2):
        """
        Return a delta quaternion ``dq`` such that ``q2 = self * dq`` where ``q2``
        is anything that instantiates a ``Quat`` object.

        This method returns the delta quaternion which represents the transformation
        from the frame of this quaternion (``self``) to ``q2``.

          q = Quat(numpy.dot(q1.transform, q2.transform))

        Example usage::

          >>> q1 = Quat((20, 30, 0))
          >>> q2 = Quat((20, 30.1, 1))
          >>> dq = q1.dq(q2)
          >>> dq.equatorial
          array([  1.79974166e-15,   1.00000000e-01,   1.00000000e+00])
        """
        if not isinstance(q2, Quat):
            q2 = Quat(q2)
        return self.inv() * q2


def normalize(array):
    """
    Normalize a 4 element array/list/numpy.array for use as a quaternion

    :param quat_array: 4 element list/array
    :returns: normalized array
    :rtype: numpy array

    """
    quat = np.array(array)
    return quat / np.sqrt(np.dot(quat, quat))
