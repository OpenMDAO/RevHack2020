import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

class pyNBSolver:
    def __init__(self, \
                 M  = [[1.0]], \
                 C  = None, \
                 K  = [[1.0]], \
                 F  = None, \
                 dF = None, \
                 N  = 10, \
                 Ni = 10, \
                 dt = 0.01, \
                 d0 = [0.0], \
                 v0 = [0.0], \
                 gm = 0.5, \
                 bt = 0.25, \
                 sl = 'full', \
                 ft = 'hdf', \
                 lm = 100.0):
        # Control parameters
        self.M  = np.array(M)
        if C is None:
            self.C = np.zeros_like(self.M)
        else:
            self.C  = np.array(C)
        self.K  = np.array(K)
        self.dF = dF
        if F is None:
            self.F = lambda t: np.zeros((len(self.M)))
        else:
            self.F  = F
        self.N  = N
        self.Ni = Ni
        self.dt = dt
        self.gm = gm
        self.bt = bt
        self.sl = sl
        self.ft = ft
        self.dm = len(self.M)
        if isinstance(lm, float):
            self.lm = abs(lm)*np.ones((self.dm))
        else:
            self.lm = np.abs(lm)

        # Scheme constants
        self.p1 = (1.0-self.gm)*self.dt
        self.p2 = self.gm*self.dt
        self.p3 = self.dt
        self.p4 = 0.5*(1.0-2.0*self.bt)*self.dt**2
        self.p5 = self.bt*self.dt**2
        self.p6 = 1.0/self.p5
        self.p7 = self.p6*self.gm
        self.p8 = self.gm/(self.bt*self.dt)
        self.Mm = self.p6*(self.M + self.p2*self.C)
        self.ep = 1e-6

        # System constants
        w,v = np.linalg.eig(np.dot(np.linalg.inv(self.M),self.K))
        self.wn = np.sqrt(w)
        self.ph = v

        # Check data output file type
        if self.ft == 'csv':
            self.saveData = self._saveData_csv
            self.loadData = self._loadData_csv
        else:
            self.saveData = self._saveData_hdf
            self.loadData = self._loadData_hdf

        # Initialization
        self.initSol(d0, v0)

    def initSol(self, d0, v0):
        self.ts = np.zeros((self.N+1))
        self.d  = np.zeros((self.N+1,self.dm))
        self.v  = np.zeros((self.N+1,self.dm))
        self.a  = np.zeros((self.N+1,self.dm))
        self.f  = np.zeros((self.N+1,self.dm))
        self.d[0] = np.array(d0)
        self.v[0] = np.array(v0)

        # Initialize at step n=0
        # Linear force
        if self.dF is None:
            self.f[0] = self.F(0)
        # Nonlinear force
        else:
            self.f[0] = self.F(d0, v0, 0)
        self.a[0] = np.linalg.solve(self.M, \
                                    self.f[0] \
                                    - np.dot(self.C,self.v[0]) \
                                    - np.dot(self.K,self.d[0]))
        # For partitioned solver, estimate step n=1
        if self.sl == 'part':
            self.v[1] = self.v[0] + self.dt*self.a[0]
            self.d[1] = self.d[0] + self.dt*self.v[0] + 0.5*self.dt**2*self.a[0]
            self.f[1] = self.F(self.d[1], self.v[1], self.dt)
            self.a[1] = np.linalg.solve(self.M, \
                                        self.f[1] \
                                        - np.dot(self.C,self.v[1]) \
                                        - np.dot(self.K,self.d[1]))
            self.ts[1] = self.dt
            self.nCur  = 2
        # For fully-coupled solver, no more initialization
        elif self.sl == 'full':
            self.nCur = 1
        # For MD-coupling, initialization done by external routines
        elif self.sl == 'cpld':
            self.nCur = 0
        else:
            print('Unknown scheme option')

# --------------------------
# Solver
# --------------------------
    def nbStep(self, t):
        # Aux variables
        n  = self.nCur
        self.ts[n] = t
        vn = self.v[n-1] + self.p1*self.a[n-1]
        dn = self.d[n-1] + self.p3*self.v[n-1] + self.p4*self.a[n-1]

        # Modified matrices
        Kn = self.Mm + self.K
        Fc = np.dot(self.Mm,dn) - np.dot(self.C,vn)

        # Update data
        if self.sl == 'full':
            if self.dF is None:
                Ft        = self.F(t)
                self.d[n] = np.linalg.solve(Kn, Ft+Fc)
            else:
                self.d[n], Ft = self._NRsolve(t, self.d[n-1], Kn, Fc, dn, vn)
        else:
            Ft        = 2.0*self.f[n-1] - self.f[n-2]
            self.d[n] = np.linalg.solve(Kn, Ft+Fc)
        self.a[n] = self.p6*(self.d[n]-dn)
        self.v[n] = vn + self.p2*self.a[n]
        if self.sl == 'full':
            self.f[n] = Ft
        else:
            self.f[n] = self.F(self.d[n], self.v[n], t)

    def stepCounter(self):
        self.nCur = self.nCur + 1
        self.ts[self.nCur] = self.nCur*self.dt
        return self.ts[self.nCur], self.nCur

    def timeStepping(self):
    # Interface for pyAEProblem
        # Step counter
        n  = self.nCur

        # Modified quantities
        vn = self.v[n-1] + self.p1*self.a[n-1]
        dn = self.d[n-1] + self.p3*self.v[n-1] + self.p4*self.a[n-1]
        Kn = self.Mm + self.K
        Fc = np.dot(self.Mm,dn) - np.dot(self.C,vn)

        # Update solution
        Ft        = self.F(0.0)
        self.d[n] = np.linalg.solve(Kn, Ft+Fc)
        self.a[n] = self.p6*(self.d[n]-dn)
        self.v[n] = vn + self.p2*self.a[n]
        self.f[n] = Ft

        # Sanity check
        if np.sum(np.abs(self.d[n])>self.lm) > 0:
            print('Divergence')
            return 0
        else:
            return 1

    def _NRsolve(self, t, di, Kn, Fc, dn, vn):
        d0  = di.copy()
        dd  = np.ones_like(d0)
        idx = 0
        while np.sum(np.abs(dd)) > self.ep and idx < self.Ni:
            v0  = vn+self.p8*(d0-dn)
            Ft  = self.F(d0,v0,t)
            R   = np.dot(Kn,d0)-(Ft+Fc)
            jac = self.dF(d0,v0,t)
            dR  = Kn - (jac[0]+self.p8*jac[1])
            dd  = np.linalg.solve(dR, -R)
            d0  = d0+dd
            idx = idx+1
        v0 = vn+self.p8*(d0-dn)
        Ft = self.F(d0,v0,t)
        return d0, Ft

    def solver(self):
        tmp = self.nCur
        for idx in range(tmp,self.N+1):
            self.nCur = idx
            self.nbStep(self.dt*idx)
            if np.sum(np.abs(self.d[self.nCur])>self.lm) > 0:
                print('Divergence')
                break
        self.trimSol()

    def trimSol(self):
        if self.nCur < self.N:
            self.N  = self.nCur
            self.ts = self.ts[:self.nCur]
            self.d  = self.d[:self.nCur,:]
            self.v  = self.v[:self.nCur,:]
            self.a  = self.a[:self.nCur,:]
            self.f  = self.f[:self.nCur,:]

# --------------------------
# Getters and setters for pyAEProblem
# --------------------------
    def setForces(self, force):
        self.F = lambda t: force
        self.f[self.nCur] = force

    def getCurDispl(self):
        return self.d[self.nCur]

    def getCurVeloc(self):
        return self.v[self.nCur]

    def getCurAccel(self):
        return self.a[self.nCur]

    def setCurAccel(self):
        n         = self.nCur
        self.a[n] = np.linalg.solve(self.M, self.f[n] \
                                      - np.dot(self.C,self.v[n]) \
                                      - np.dot(self.K,self.d[n]))

    def setNewState(self):
        n         = self.nCur
        self.v[n] = self.v[n-1] + self.dt*self.a[n-1]
        self.d[n] = self.d[n-1] + self.dt*self.v[n-1] + 0.5*self.dt**2*self.a[n-1]

# --------------------------
# Post-processing
# --------------------------
    def systIden(self, Nm=2):
        wm = np.max(self.wn)            # Maximum natural frequency of system
        Te = (2.0*np.pi/wm)/4.0*0.97    # Sampling frequency / period. Not sure what the 0.97 is
        kt = int(Te/self.dt)            # Represents the sampling interval
        tn = np.arange(0,self.N,kt)     # Pick the kt timestep numbers (N = #timesteps). K-th point in discrete time.
        xs = self.d[tn,0]               # Pick out displacements tn location. Essentially downsamlping the actual displacement
        Nt = len(xs)                    # Number of samples we downsampled
        Nc = 2*Nm                       # Number of AR coefficients. Nm = number of modes of system.  Each mode will give its damping and frequency.

        # Set up least squares problem. Problem is overdetermined.
        rh = -xs[Nt:Nc-1:-1]            # Right hand side.
        Ah = np.zeros((Nt-Nc,Nc))       #
        for idx in range(Nt-Nc):
            Ah[idx] = xs[-2-idx:-2-idx-Nc:-1]

        try:
            an = np.linalg.solve(np.dot(Ah.T,Ah), np.dot(Ah.T,rh))

            Ap  = np.eye(Nc, k=1)
            Ap[:,0] = -an
            w,v = np.linalg.eig(Ap)
            r   = np.real(w)
            s   = np.imag(w)
            w0  = np.arctan(np.abs(s/r))/Te
            z0  = 0.5*np.log(s*s+r*r)/Te / w0

            idx = ~(np.isnan(w0) | np.isinf(w0))
            w   = w0[idx]
            z   = z0[idx]
            idx = np.argsort(w)
            w   = w[idx[::2]]/(2.*np.pi)
            z   = z[idx[::2]]
        except:
            # Probably diverged
            w = np.array([0.0, 0.0])
            z = 1.0/w

        return np.max(z), w, z

    def pltSol(self, leg=[], xlbl=''):
        sty = ['b-','k-','r-','g-','b--','k--','r--','g--']
        if leg == []:
            for idx in range(self.dm):
                leg.append('Var'+str(idx))
        if xlbl == '':
            xlbl = 't, s'
        f = plt.figure()
        for idx in range(self.dm):
            plt.plot(self.ts,self.d[:,idx],sty[np.mod(idx,8)],label=leg[idx])
        plt.xlabel(xlbl)
        plt.legend(loc=0)
        plt.grid()
        return f

    def _loadData_hdf(self, fname):
        f = h5py.File(fname,'r')

        self.N  = f["csd/para/N"][()]
        self.dm = f["csd/para/dm"][()]
        self.dt = f["csd/para/dt"][()]
        self.ts = f["csd/time"][()]
        self.d  = f["csd/disp"][()]
        self.v  = f["csd/velo"][()]
        self.a  = f["csd/acce"][()]
        self.f  = f["csd/forc"][()]

    def _loadData_csv(self, dat):
        dim     = self.dm
        self.ts = np.array(dat[0])
        self.N  = len(self.ts)-1
        self.dt = self.ts[1]-self.ts[0]
        self.d  = np.array(dat[1      :1+dim  ]).transpose()
        self.v  = np.array(dat[1+dim  :1+2*dim]).transpose()
        self.a  = np.array(dat[1+2*dim:1+3*dim]).transpose()
        self.f  = np.array(dat[1+3*dim:1+4*dim]).transpose()

    def _saveData_hdf(self, name='res', output='./'):
        fname = output+name+'.hdf5'
        try:
            f = h5py.File(fname, "r+")
        except:
            f = h5py.File(fname, "w")
        try:
            zm, w, z = self.systIden(self.dm)
        except:
            print('Error in SI')
            zm = -1.0
            w  = [0.0,0.0]
            z  = [-1.0,-1.0]
        f.create_dataset("csd/para/dm", data=self.dm)
        f.create_dataset("csd/para/N",  data=self.N)
        f.create_dataset("csd/para/dt", data=self.dt)
        f.create_dataset("csd/para/zm", data=zm)
        f.create_dataset("csd/para/om", data=w)
        f.create_dataset("csd/para/ze", data=z)
        f.create_dataset("csd/time", data=self.ts)
        f.create_dataset("csd/disp", data=self.d)
        f.create_dataset("csd/velo", data=self.v)
        f.create_dataset("csd/acce", data=self.a)
        f.create_dataset("csd/forc", data=self.f)

    def _saveData_csv(self, name='res', output='./'):
        fName = os.path.join(output, name)
        data = np.hstack((self.ts.reshape(self.N+1,1), self.d, self.v, self.a, self.f))
        #np.savetxt(output+name+'.csv', data.transpose(), delimiter=', ')
        np.savetxt(fName, data.transpose(), delimiter=', ')

## Test suites
## Case 1: Linear with damping
# F   = lambda t: np.array([0.1,1.0])
# an1 = lambda t: 0.9*np.exp(-t)*(1.0+t)+0.1
# k1  = np.sqrt(3)/2.0
# an2 = lambda t: 1-np.exp(-0.5*t)/k1*np.sin(k1*t+np.pi/3.0)
# sol = pynb.pyNBSolver(\
#                 M=[[1.0,0.0],[0.0,1.0]],\
#                 C=[[2.0,0.0],[0.0,1.0]],\
#                 K=[[1.0,0.0],[0.0,1.0]],\
#                 F=F,d0=[1.0,0.0],v0=[0.0,0.0],N=1000,dt=0.02)
# sol.solver()
# sol.pltSol()
# plt.plot(sol.ts,an1(sol.ts), 'b--')
# plt.plot(sol.ts,an2(sol.ts), 'k--')
# plt.show()

## Case 2: Nonlinear
# F   = lambda d, v, t: np.array([0.1*d[1]**2,np.sin(d[0])])
# dF  = lambda d, v, t: [np.array([[0.0, 0.2*d[1]], [np.cos(d[0]), 0.0]]),\
#                        np.zeros((2,2))]
# sol = pynb.pyNBSolver(\
#                 M=[[1.0,0.0],[0.0,1.0]],\
#                 C=[[0.1,0.0],[0.0,0.2]],\
#                 K=[[1.0,0.0],[0.0,1.0]],\
#                 F=F,dF=dF,\
#                 d0=[1.0,0.0],v0=[0.0,0.0],\
#                 N=1000,dt=0.05)
# sol.solver()
# sol.saveData()
#
# sol = NDSolve[{
#     x''[t] + x'[t]/10 + x[t] == y[t]^2/10,
#     y''[t] + y'[t]/5 + y[t] == Sin[x[t]],
#     x[0] == 1, x'[0] == 0, y[0] == 0, y'[0] == 0}, {x[t], y[t]}, {t,
#     0, 50}];
# f1 = Plot[Evaluate@{x[t], y[t]} /. sol, {t, 0, 50}];
# SetDirectory[NotebookDirectory[]];
# dat = Import["res.csv"];
# f2 = ListPlot[{dat[[All, {1, 2}]], dat[[All, {1, 3}]]}];
# Show[f1, f2]
