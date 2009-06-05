from Quaternion import Quat
ra = 10.
dec = 20.
roll = 30.

print '-' * 40
print

q_eq = Quat([ra, dec, roll])
print 'q_eq.equatorial=', q_eq.equatorial
print 'q_eq.q=', q_eq.q
print 'q_eq.transform=', q_eq.transform
print
q_eq._equatorial = None
q_eq._T = None
print 'q_eq.transform=', q_eq.transform
print

q1, q2, q3, q4 = q_eq.q

# q_eq_inv = q_eq.inv()
q_eq_inv = Quat([ 340.99173674 , -11.82213076 , 326.246305  ])
print 'q_eq_inv.equatorial=', q_eq_inv.equatorial
# print 'q_eq_inv.q=', q_eq_inv.q
print 'q_eq_inv.transform=', q_eq_inv.transform
print
print q_eq_inv.equatorial

print '(q_eq * q_eq_inv).q=', (q_eq * q_eq_inv).q

q_eq_inv = q_eq.inv()
print '(q_eq * q_q_inv).q=', (q_eq * q_eq_inv).q


