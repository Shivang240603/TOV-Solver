from math import pi, log, sqrt
from scipy.integrate import odeint
from numpy import arange, zeros, linspace
from matplotlib.pylab import plot, scatter
from sympy import symbols, Eq, solve

#Equation of State
#     P = d^gamma
#P is pressure, d is density
#Returns the pressure based on the density

def eos(pressure):
  return pressure**0.5+pressure

#This returnd dP/dr and dm/dr based on input variables
#Vars is in the form of [density, mass], and a radius value

def tov(vars, radius):
  density = eos(vars[0])
  mass = vars[1]
  pres = vars[0]
  dPdr = -1*(((density+pres)*(mass+(4*pi*pres*radius**3)))/(radius*(radius-(2*mass))))
  dmdr = 4*pi*density*radius**2
  return dPdr, dmdr

#tovSolve uses a starting radius and a final radius and distributes posible radius values between them
#The odeint function is used to solve the TOV equations using the initial values for the pressure and mass
#Initial pressure is just the density run through the EoS
#Initial mass is the density times the volume ((4/3)*pi*r^3) where r is the starting radius
#Then, as long as the pressure is above 1, the star is considered to exist
#Once the pressure has dropped below that, the star has ended and that final mass and radius are returned, along with the pressure

def tovSolve(pressure):
  startingRadius =0.1
  finalRadius = 1500
  step = 0.1

  ans = odeint(tov, [pressure, 0],arange(startingRadius, finalRadius, step), printmessg=1)
  print("ANS ===>", ans)
  radii = arange(startingRadius, finalRadius, step)
  masses = ans[:,1]
  pressures = ans[:,0]

  count = 0
  mass = 0.0
  pressure = 0.0
  radius = startingRadius

  for i in pressures:
    if i > 1e-13*pressure:
      count += 1
      mass = masses[count]
      radius += step
  return radius, mass

#This function takes in the minimum and maximum density and the number of density wanted to be check
#Each density is then run through tovSolve, and the resulting values are saved in lists

def mass_radius(minimumDensity, maximumDensity, step):
  dens = zeros(step)
  mass = zeros(step)
  radius = zeros(step)
  pressure = zeros(step)
  for i in range(step):
    dens[i] = minimumDensity + (maximumDensity-minimumDensity)*i/step
    radius[i],mass[i] = tovSolve(dens[i])
    print(i , "=====", dens[i], " ====== ", radius[i])
  return radius, mass, dens

ans = mass_radius(5e-5, 150, 100)
plot(ans[0], ans[1])
