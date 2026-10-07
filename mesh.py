import numpy as np

# class containing mesh related data
class Mesh(object):
    def __init__(self, n=None, center=0, size=1, gamma=1.4, cs=1):
        self.t = 0.0                           # time
        self.dt = 0.0                          # time step
        self.step = 0                          # number of updates (integer)
        self.n = n                             # resolution of the grid
        self.ndim = len(n)                     # dimensionality of the grid
        self.gamma = gamma                     # adiabatic index, currently an isothermal equation of state is not supported
        self.cs    = cs                        # sound speed only relevant if we support gamma=1.
        self.isothermal = (gamma == 1)         # isothermal equation of state  

        self.nv = 1 + (not self.isothermal) + self.ndim # number of variables
        # Init data to NaN to promote explicitly setting initial conditions
        self.vars = np.full((self.nv,)+self.n, np.nan) # dynamic variables
        self.iD = 0
        if self.isothermal:                    # if isothermal, energy is not an independent variable
            self.varnames=(['D','vx','vy','vz'])[0:1+self.ndim]
            self.iE = None
            self.iM = slice(1,self.nv)         # slice for Numpy expressions
            self.iMl = np.array(range(1,self.nv))  # list for referencing components
        else:
            self.varnames=(['D','E','vx','vy','vz'])[0:2+self.ndim]
            self.iE = 1
            self.E = self.vars[self.iE]        # total energy
            self.iM = slice(2,self.nv)         # slice for Numpy expressions
            self.iMl = np.array(range(2,self.nv))  # list for referencing components
        self.D = self.vars[self.iD]            # density
        self.M = self.vars[self.iM]            # momentum

        # make cell centered coordinates
        self.ds = np.array(size)/n             # cell size
        self.size = self.ds*n                  # domain size
        self.lb = center - 0.5*self.size       # lower bound of the domain
        self.ub = center + 0.5*self.size       # upper bound of the domain

        self.axis = []                         # cell centered 1D coordinates
        for i in range(self.ndim):
            self.axis.append(np.linspace(self.lb[i],self.ub[i],self.n[i], endpoint=False) + 0.5 * self.ds[i])

        self.x = self.axis[0]
        self.coords = np.meshgrid(*self.axis,indexing='ij',sparse=True) # expand to 3D grid
        if self.ndim >= 2:
            self.y = self.axis[1]
            self.rcyl = np.sqrt(self.coords[0]**2 + self.coords[1]**2) # cylindrical radius
            self.r = self.rcyl
        if self.ndim == 3:
            self.z = self.axis[2]
            self.rsph = np.sqrt(self.coords[0]**2 + self.coords[1]**2 + self.coords[2]**2) # spherical radius
            self.r = self.rsph

    def velocity(self):                            # velocity from momentum and density
        return self.M / self.D[None]

    def pressure(self):                            # gas pressure
        if self.isothermal: return self.cs**2*self.D
        Ekin = self.M[0]**2
        for i in range(1,self.ndim):
            Ekin += self.M[i]**2
        Ekin /= 2*self.D
        return (self.gamma-1)*(self.E - Ekin)

    def v_sound(self):                             # sound speed
        if self.isothermal: return self.cs
        return np.sqrt(self.gamma*self.pressure()/self.D)

    def temperature(self):                         # temperature of gas per mu/kB
        if self.isothermal: return self.cs**2
        return self.pressure() / self.D