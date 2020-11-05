import numpy as np
import matplotlib.pylab as plt

import openmdao.api as om

cr = om.CaseReader('cases.sql')
cases = cr.get_cases()
n = len(cases)

alpha = np.zeros(n)
horiz_tail_area = np.zeros(n)
vert_tail_area = np.zeros(n)
wing_cord = np.zeros(n)

CL = np.zeros(n)
con_alpha = np.zeros(n)
l_over_d = np.zeros(n)
CN_beta = np.zeros(n)
L_equals_W = np.zeros(n)

for n, case in enumerate(cases):
    alpha[n] = case['ecrm_150.alpha']
    #horiz_tail_area[n] = case['horiz_tail_area']
    #vert_tail_area[n] = case['vert_tail_area']
    wing_cord[n] = case['wing_cord']

    CL[n] = case['ecrm_150.CL']
    con_alpha[n] = case['con_alpha_150.val']
    l_over_d[n] = case['l_over_d.val']
    CN_beta[n] = case['ecrm_150.CN_beta']
    L_equals_W[n] = case['ecrm_150.L_equals_W']

plt.figure(0)
plt.plot(alpha, CL, '+')
plt.figure(1)
plt.plot(alpha, L_equals_W, '+')
plt.figure(2)
plt.plot(alpha, l_over_d, '+')
plt.figure(3)
plt.plot(alpha, con_alpha, '+')
plt.figure(4)
plt.plot(alpha, CN_beta, '+')
plt.show()
print('done')

# Constraint: -CM_α/CL_α > 0.0
# Constraint: CN_β > 0.0
# Constraint: CL < 1.3
# Constraint: CL = W/qS