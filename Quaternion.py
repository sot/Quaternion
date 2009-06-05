"""
Quaternion provides a class, Quat, and its methods.  This class provides:

   - a convenient constructor to convert to/from (RA,Dec,Roll) to/from quaternions
   - class methods to multiply and divide quaternions 
"""


      
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
    >>> q2 = Quat([ 0.38857298, -0.3146602 ,  0.23486498,  0.8335697])
    >>> q2.ra
    11.999999315925008


   __mul__ and __div__ are overloaded for the class to perform quaternion
   multiplication and division

   Example usage::
   
    >>> q1 = Quat((20,30,40))
    >>> q2 = Quat((30,40,50))
    >>> q = q1 / q2

   Performs the operation as q1 * inverse q2

   Example usage::

    >>> q1 = Quat((20,30,40))
    >>> q2 = Quat((30,40,50))
    >>> q = q1 * q2


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
         elif attitude.shape == (3,3):
            self._set_transform(attitude)
         elif attitude.shape == (3,):
            self._set_equatorial(attitude)
         else:
            raise TypeError("attitude is not one of possible types (3 or 4 elements, Quat, or 3x3 matrix)")
         

   def _set_q(self, q):
      """ set the value of the 4 element quaternion vector """
      q = np.array(q)
      if abs(np.sum(q**2) - 1.0) > 1e-6:
         raise ValueError('Quaternion must be normalized so sum(q**2) == 1; use Quaternion.normalize')
      self._q = q
      # Erase internal values of other representations
      self._equatorial = None
      self._T = None

   def _get_q(self):
      """
      Retrieve value of q
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

   def _set_equatorial(self, equatorial):
      """set the value of the 3 element equatorial coordinate list [RA,Dec,Roll]
         expects values in degrees
         bounds are not checked
      """
      att = np.array(equatorial)
      ra, dec, roll = att
      self._ra0 = ra
      if ( ra > 180 ):
         self._ra0 = ra - 360
         self._roll0 = roll
      if ( roll > 180):
         self._roll0 = roll - 360
      self._equatorial = att
    
   def _get_equatorial(self):
      """retrieve [RA, Dec, Roll]"""
      if self._equatorial is None:
         if self._q is not None:
            self._equatorial = self._quat2equatorial()
         elif self._T is not None:
            self._q = self._transform2quat()
            self._equatorial = self._quat2equatorial()
      return self._equatorial

   equatorial = property(_get_equatorial,_set_equatorial)

   def _get_ra(self):
      """retrieve RA term from equatorial system in degrees"""
      return self.equatorial[0]
        
   def _get_dec(self):
      """retrieve Dec term from equatorial system in degrees"""
      return self.equatorial[1]
        
   def _get_roll(self):
      """retrieve Roll term from equatorial system in degrees"""
      return self.equatorial[2]

   ra = property(_get_ra)
   dec = property(_get_dec)
   roll = property(_get_roll)

   def _set_transform(self, T):
      """set the value of the 3x3 rotation/transform matrix"""
      transform = np.array(T)
      self._T = transform

   def _get_transform(self):
      """retrieve the value of the 3x3 rotation/transform matrix"""
      if self._T is None:
         if self._q is not None:
            self._T = self._quat2transform()
         elif self._equatorial is not None:
            self._T = self._equatorial2transform()
      return self._T

   transform = property(_get_transform, _set_transform)

   def _quat2equatorial(self):
      """ determine RA, Dec, and Roll for the object quaternion"""
      q = self.q
      q2 = self.q**2

      ## calculate direction cosine matrix elements from $quaternions
      xa = q2[0] - q2[1] - q2[2] + q2[3] 
      xb = 2 * (q[0] * q[1] + q[2] * q[3]) 
      xn = 2 * (q[0] * q[2] - q[1] * q[3]) 
      yn = 2 * (q[1] * q[2] + q[0] * q[3]) 
      zn = q2[3] + q2[2] - q2[0] - q2[1] 

      ##; calculate RA, Dec, Roll from cosine matrix elements
      ra   = degrees(atan2(xb , xa)) ;
      dec  = degrees(atan2(xn , sqrt(1 - xn**2)));
      roll = degrees(atan2(yn , zn)) ;
      if ( ra < 0 ):
         ra += 360
      if ( roll < 0 ):
         roll += 360

      return np.array([ra, dec, roll])


   def _quat2transform(self):
      """
      Transform a unit quaternion into its corresponding rotation matrix (to
      be applied on the right side).
      
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
      rmat[0,0] = 1. - yy2 - zz2
      rmat[0,1] = xy2 - wz2
      rmat[0,2] = zx2 + wy2
      rmat[1,0] = xy2 + wz2
      rmat[1,1] = 1. - xx2 - zz2
      rmat[1,2] = yz2 - wx2
      rmat[2,0] = zx2 - wy2
      rmat[2,1] = yz2 + wx2
      rmat[2,2] = 1. - xx2 - yy2
      
      return rmat

   def _equatorial2quat( self ):
      """ construct the object 4 quat if needed """
      return self._transform2quat()
   
   def _equatorial2transform( self ):
      """ construct the transform/rotation matrix from RA,Dec,Roll"""
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
      rmat = np.array([[ca * cd,                    sa * cd,                  sd     ],
                       [-ca * sd * sr - sa * cr,   -sa * sd * sr + ca * cr,   cd * sr],
                       [-ca * sd * cr + sa * sr,   -sa * sd * cr - ca * sr,   cd * cr]])

      return rmat.transpose()

   def _transform2quat( self ):
      """ determine the quat from the transform/rotation matrix """
      # Code was copied from perl PDL code that uses backwards index ordering
      T = self.transform.transpose()  
      den = np.array([ 1.0 + T[0,0] - T[1,1] - T[2,2],
                      1.0 - T[0,0] + T[1,1] - T[2,2],
                      1.0 - T[0,0] - T[1,1] + T[2,2],
                      1.0 + T[0,0] + T[1,1] + T[2,2]])
      
      match = np.flatnonzero( den == max(den) )
      if ( len(match) > 1 ):
         raise ValueError( "Extra matches")
      max_idx = int(match)

      q = np.zeros(4)
      q[max_idx] = 0.5 * sqrt( max(den) )
      denom = 4.0 * q[max_idx]
      if (max_idx == 0):
         q[1] =  (T[1,0] + T[0,1]) / denom 
         q[2] =  (T[2,0] + T[0,2]) / denom 
         q[3] = -(T[2,1] - T[1,2]) / denom 
      if (max_idx == 1):
         q[0] =  (T[1,0] + T[0,1]) / denom 
         q[2] =  (T[2,1] + T[1,2]) / denom 
         q[3] = -(T[0,2] - T[2,0]) / denom 
      if (max_idx == 2):
         q[0] =  (T[2,0] + T[0,2]) / denom 
         q[1] =  (T[2,1] + T[1,2]) / denom 
         q[3] = -(T[1,0] - T[0,1]) / denom 
      if (max_idx == 3):
         q[0] = -(T[2,1] - T[1,2]) / denom 
         q[1] = -(T[0,2] - T[2,0]) / denom 
         q[2] = -(T[1,0] - T[0,1]) / denom 

      if (q[3] < 0):
         q = -q

      return q


   def __div__(self, quat2):
      """
      Divide one quaternion by another.
      
      Example usage::

       >>> q1 = Quat((20,30,40))
       >>> q2 = Quat((30,40,50))
       >>> q = q1 / q2

      Performs the operation as q1 * inverse q2

      :rtype: Quat

      """
      return self * quat2.inv()


   def __mul__(self, quat2):
      """
      Multiply quaternion by another.

      Example usage::

       >>> q1 = Quat((20,30,40))
       >>> q2 = Quat((30,40,50))
       >>> q = q1 * q2

       :rtype: Quat

      """

      q1 = self.q
      q2 = quat2.q
      mult = np.zeros(4)
   ## x = w1*x2 - z1*y2 + y1*z2 + x1*w2
   ## y = z1*x2 + w1*y2 - x1*z2 + y1*w2
   ## z = y1*x2 + x1*y2 + w1*z2 + z1*w2
   ## w = x1*x2 - y1*y2 - z1*z2 + w1*w2
      mult[0] = q1[3]*q2[0] - q1[2]*q2[1] + q1[1]*q2[2] + q1[0]*q2[3]
      mult[1] = q1[2]*q2[0] + q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3]
      mult[2] = -q1[1]*q2[0] + q1[0]*q2[1] + q1[3]*q2[2] + q1[2]*q2[3]
      mult[3] = -q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] + q1[3]*q2[3]
      q = mult.copy()
      if (mult[3] < 0):
         q = -mult
      return Quat(q)


   def inv(self):
      """
      Invert the quaternion 
      
      :rtype: Quat
      """
      return Quat([self.q[0], self.q[1], self.q[2], -self.q[3]])


def normalize(array):
   """ 
   Normalize a 4 element array/list/numpy.array for use as a quaternion
   
   :param quat_array: 4 element list/array
   
   :rtype: numpy array

   """
   quat = np.array(array)
   return quat / np.sqrt(np.dot(quat, quat))


