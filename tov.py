import os
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np
import math

# Constants
G = 6.6730831e-8
c = 2.99792458e10
MeV_fm3_to_pa = 1.6021766e35
c_km = 2.99792458e5  # km/s
mN = 1.67e-24  # g
mev_to_ergs = 1.602176565e-6
fm_to_cm = 1.0e-13
ergs_to_mev = 1.0 / mev_to_ergs
cm_to_fm = 1.0 / fm_to_cm
Msun = 1.988435e33
MeV_fm3_to_pa_cgs = 1.6021766e33
km_to_mSun = G / c ** 2
hbarc3 = 197.32700288295746 ** 3
nucleon_mass = 938.04
pi = math.pi


class TOV:
    """
    Instance of the TOV solver
    """

    def __init__(self, en_arr, p_arr, add_crust=True, plot_eos=False):
        print("Initializing TOV solver...")
        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        en_arr *= MeV_fm3_to_pa_cgs / c ** 2
        p_arr *= MeV_fm3_to_pa_cgs
        print("Energy array:", en_arr)
        print("Pressure array:", p_arr)

        sort_ind = np.argsort(p_arr)
        self.en_dens = interp1d(p_arr[sort_ind], en_arr[sort_ind], kind='cubic', bounds_error=False,
                                fill_value=(en_arr[sort_ind][0], en_arr[sort_ind][-1]))

        sort_ind = np.argsort(en_arr)
        self.press = interp1d(en_arr[sort_ind], p_arr[sort_ind], kind='cubic', bounds_error=False,
                              fill_value=(p_arr[sort_ind][0], p_arr[sort_ind][-1]))

        self.__en_arr = en_arr
        self.__p_arr = p_arr

        self.min_dens = np.min(en_arr)
        self.max_dens = np.max(en_arr)

        self.min_p = np.min(p_arr)
        self.max_p = np.max(p_arr)

        if add_crust:
            if plot_eos:
                plt.plot(self.__en_arr / (MeV_fm3_to_pa_cgs / c ** 2),
                         self.__p_arr / MeV_fm3_to_pa_cgs,
                         linestyle='-', label='original EOS')
            self.add_crust()
        if plot_eos:
            plt.plot(self.__en_arr / (MeV_fm3_to_pa_cgs / c ** 2),
                     self.__p_arr / MeV_fm3_to_pa_cgs,
                     linestyle='--', label='EOS with crust')

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'${\rm \varepsilon~(MeV/fm^{3}) }$')
            plt.ylabel(r'${\rm P~(MeV/fm^{3}) }$')
            plt.legend()
            plt.show()

    def add_crust(self):
        """
        Adds Nuclear Statistical Equilibrium crust EOS.
        """
        print("Adding crust...")
        dir_name = os.path.dirname(__file__)
        crust_loc = os.path.join(dir_name, 'data', 'Baym_eos.dat')

        baym_eos = np.genfromtxt(crust_loc, dtype=float, skip_header=1, names=["en", "p", "nB"])

        P_crust = interp1d(baym_eos["en"], baym_eos["p"], kind='cubic', bounds_error=False,
                           fill_value=(baym_eos["p"][0], baym_eos["p"][-1]))

        def eq_glue(n):
            return P_crust(n) - self.press(n)

        g = root(eq_glue, [44. * (MeV_fm3_to_pa_cgs / c ** 2)], options={'maxfev': 200})

        n_glue = g['x'][0]

        en_arr = []
        p_arr = []

        for i in range(len(baym_eos["p"])):
            if baym_eos["en"][i] < n_glue:
                en_arr.append(baym_eos["en"][i])
                p_arr.append(baym_eos["p"][i])
            else:
                break

        for i in range(len(self.__p_arr)):
            if self.__en_arr[i] >= n_glue:
                en_arr.append(self.__en_arr[i])
                p_arr.append(self.__p_arr[i])

        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        self.min_dens = min(en_arr)
        self.min_p = min(p_arr)

        self.en_dens = interp1d(p_arr, en_arr, kind='cubic', bounds_error=False, fill_value=(en_arr[0], en_arr[-1]))
        self.press = interp1d(en_arr, p_arr, kind='cubic', bounds_error=False, fill_value=(p_arr[0], p_arr[-1]))

        self.__en_arr = en_arr
        self.__p_arr = p_arr

    def dedp(self, r, R_dep):
        e_R, p_R, m_R = R_dep

        p = p_R(r)
        dp = p * 0.005

        # Ensure values are within bounds
        p_minus_3dp = max(p - 3 * dp, self.min_p)
        p_minus_2dp = max(p - 2 * dp, self.min_p)
        p_minus_1dp = max(p - 1 * dp, self.min_p)
        p_plus_3dp = min(p + 3 * dp, self.max_p)
        p_plus_2dp = min(p + 2 * dp, self.max_p)
        p_plus_1dp = min(p + 1 * dp, self.max_p)

        el_3 = self.en_dens(p_minus_3dp)
        el_2 = self.en_dens(p_minus_2dp)
        el_1 = self.en_dens(p_minus_1dp)
        er_3 = self.en_dens(p_plus_3dp)
        er_2 = self.en_dens(p_plus_2dp)
        er_1 = self.en_dens(p_plus_1dp)

        de_dp = (-1 / 60 * el_3 + 3 / 20 * el_2 - 3 / 4 * el_1 + 3 / 4 * er_1 - 3 / 20 * er_2 + 1 / 60 * er_3) / dp

        return de_dp

    def love_eq(self, param, r, R_dep):
        beta, H = param
        e_R, p_R, m_R = R_dep

        try:
            dummy = p_R(r)
        except ValueError:
            return [100000, 100000]

        de_dp = self.dedp(r, R_dep)

        dbetadr = H * (-2 * pi * G / c ** 2 * (
                5 * e_R(r) + 9 * p_R(r) / c ** 2 + de_dp * c ** 2 * (e_R(r) + p_R(r) / c ** 2)) \
                       + 3 / r ** 2 \
                       + 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1) * (
                               m_R(r) / r ** 2 * km_to_mSun + G / c ** 4 * 4 * pi * r * p_R(r)) ** 2) \
                  + beta / r * (
                          -1 + m_R(r) / r * km_to_mSun + 2 * pi * r ** 2 * G / c ** 2 * (e_R(r) - p_R(r) / c ** 2))
        dbetadr *= 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1)

        dHdr = beta
        return [dbetadr, dHdr]

    def tov_eq(self, y, r):
        P, m = y

        if P < self.min_p or P > self.max_p:
            return [0., 0.]

        eden = self.en_dens(P)

        dPdr = -G * (eden + P / c ** 2) * (m + 4.0 * pi * r ** 3 * P / c ** 2) / (r * (r - 2.0 * G * m / c ** 2))

        dmdr = 4.0 * pi * r ** 2 * eden

        return [dPdr, dmdr]

    def solve(self, central_pressure, r_stop=1.0e-8, dr=1e-4):
        r = np.arange(r_stop, 100, dr)
        y0 = [central_pressure, 0]

        sol = odeint(self.tov_eq, y0, r)

        pressure = sol[:, 0]
        mass = sol[:, 1]

        radius = r[pressure > 0][-1]
        mass_final = mass[pressure > 0][-1]

        return radius, mass_final


# Sample energy density and pressure arrays
en_arr = [1e14, 2e14, 3e14, 4e14, 5e14]  # Placeholder values
p_arr = [1e34, 2e34, 3e34, 4e34, 5e34]   # Placeholder values

# Initialize the TOV solver
tov_solver = TOV(en_arr, p_arr)

# Define the range of central densities
central_densities = np.logspace(14, 16, 50)  # Placeholder values in g/cm^3

masses = []
radii = []

# Solve the TOV equations for each central density
for density in central_densities:
    central_pressure = tov_solver.press(density)
    radius, mass = tov_solver.solve(central_pressure)
    masses.append(mass / Msun)  # Convert to solar masses
    radii.append(radius)  # Radius in km

# Plot the mass-radius curve
plt.figure(figsize=(10, 6))
plt.plot(radii, masses, label='Mass-Radius Curve')
plt.xlabel('Radius (km)')
plt.ylabel('Mass (M☉)')
plt.title('Mass-Radius Curve for Neutron Stars')
plt.legend()
plt.grid(True)
plt.show()
import os
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np
import math

# Constants
G = 6.6730831e-8
c = 2.99792458e10
MeV_fm3_to_pa = 1.6021766e35
c_km = 2.99792458e5  # km/s
mN = 1.67e-24  # g
mev_to_ergs = 1.602176565e-6
fm_to_cm = 1.0e-13
ergs_to_mev = 1.0 / mev_to_ergs
cm_to_fm = 1.0 / fm_to_cm
Msun = 1.988435e33
MeV_fm3_to_pa_cgs = 1.6021766e33
km_to_mSun = G / c ** 2
hbarc3 = 197.32700288295746 ** 3
nucleon_mass = 938.04
pi = math.pi


class TOV:
    """
    Instance of the TOV solver
    """

    def __init__(self, en_arr, p_arr, add_crust=True, plot_eos=False):
        print("Initializing TOV solver...")
        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        en_arr *= MeV_fm3_to_pa_cgs / c ** 2
        p_arr *= MeV_fm3_to_pa_cgs
        print("Energy array:", en_arr)
        print("Pressure array:", p_arr)

        sort_ind = np.argsort(p_arr)
        self.en_dens = interp1d(p_arr[sort_ind], en_arr[sort_ind], kind='cubic', bounds_error=False,
                                fill_value=(en_arr[sort_ind][0], en_arr[sort_ind][-1]))

        sort_ind = np.argsort(en_arr)
        self.press = interp1d(en_arr[sort_ind], p_arr[sort_ind], kind='cubic', bounds_error=False,
                              fill_value=(p_arr[sort_ind][0], p_arr[sort_ind][-1]))

        self.__en_arr = en_arr
        self.__p_arr = p_arr

        self.min_dens = np.min(en_arr)
        self.max_dens = np.max(en_arr)

        self.min_p = np.min(p_arr)
        self.max_p = np.max(p_arr)

        if add_crust:
            if plot_eos:
                plt.plot(self.__en_arr / (MeV_fm3_to_pa_cgs / c ** 2),
                         self.__p_arr / MeV_fm3_to_pa_cgs,
                         linestyle='-', label='original EOS')
            self.add_crust()
        if plot_eos:
            plt.plot(self.__en_arr / (MeV_fm3_to_pa_cgs / c ** 2),
                     self.__p_arr / MeV_fm3_to_pa_cgs,
                     linestyle='--', label='EOS with crust')

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'${\rm \varepsilon~(MeV/fm^{3}) }$')
            plt.ylabel(r'${\rm P~(MeV/fm^{3}) }$')
            plt.legend()
            plt.show()

    def add_crust(self):
        """
        Adds Nuclear Statistical Equilibrium crust EOS.
        """
        print("Adding crust...")
        dir_name = os.path.dirname(__file__)
        crust_loc = os.path.join(dir_name, 'data', 'Baym_eos.dat')

        baym_eos = np.genfromtxt(crust_loc, dtype=float, skip_header=1, names=["en", "p", "nB"])

        P_crust = interp1d(baym_eos["en"], baym_eos["p"], kind='cubic', bounds_error=False,
                           fill_value=(baym_eos["p"][0], baym_eos["p"][-1]))

        def eq_glue(n):
            return P_crust(n) - self.press(n)

        g = root(eq_glue, [44. * (MeV_fm3_to_pa_cgs / c ** 2)], options={'maxfev': 200})

        n_glue = g['x'][0]

        en_arr = []
        p_arr = []

        for i in range(len(baym_eos["p"])):
            if baym_eos["en"][i] < n_glue:
                en_arr.append(baym_eos["en"][i])
                p_arr.append(baym_eos["p"][i])
            else:
                break

        for i in range(len(self.__p_arr)):
            if self.__en_arr[i] >= n_glue:
                en_arr.append(self.__en_arr[i])
                p_arr.append(self.__p_arr[i])

        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        self.min_dens = min(en_arr)
        self.min_p = min(p_arr)

        self.en_dens = interp1d(p_arr, en_arr, kind='cubic', bounds_error=False, fill_value=(en_arr[0], en_arr[-1]))
        self.press = interp1d(en_arr, p_arr, kind='cubic', bounds_error=False, fill_value=(p_arr[0], p_arr[-1]))

        self.__en_arr = en_arr
        self.__p_arr = p_arr

    def dedp(self, r, R_dep):
        e_R, p_R, m_R = R_dep

        p = p_R(r)
        dp = p * 0.005

        # Ensure values are within bounds
        p_minus_3dp = max(p - 3 * dp, self.min_p)
        p_minus_2dp = max(p - 2 * dp, self.min_p)
        p_minus_1dp = max(p - 1 * dp, self.min_p)
        p_plus_3dp = min(p + 3 * dp, self.max_p)
        p_plus_2dp = min(p + 2 * dp, self.max_p)
        p_plus_1dp = min(p + 1 * dp, self.max_p)

        el_3 = self.en_dens(p_minus_3dp)
        el_2 = self.en_dens(p_minus_2dp)
        el_1 = self.en_dens(p_minus_1dp)
        er_3 = self.en_dens(p_plus_3dp)
        er_2 = self.en_dens(p_plus_2dp)
        er_1 = self.en_dens(p_plus_1dp)

        de_dp = (-1 / 60 * el_3 + 3 / 20 * el_2 - 3 / 4 * el_1 + 3 / 4 * er_1 - 3 / 20 * er_2 + 1 / 60 * er_3) / dp

        return de_dp

    def love_eq(self, param, r, R_dep):
        beta, H = param
        e_R, p_R, m_R = R_dep

        try:
            dummy = p_R(r)
        except ValueError:
            return [100000, 100000]

        de_dp = self.dedp(r, R_dep)

        dbetadr = H * (-2 * pi * G / c ** 2 * (
                5 * e_R(r) + 9 * p_R(r) / c ** 2 + de_dp * c ** 2 * (e_R(r) + p_R(r) / c ** 2)) \
                       + 3 / r ** 2 \
                       + 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1) * (
                               m_R(r) / r ** 2 * km_to_mSun + G / c ** 4 * 4 * pi * r * p_R(r)) ** 2) \
                  + beta / r * (
                          -1 + m_R(r) / r * km_to_mSun + 2 * pi * r ** 2 * G / c ** 2 * (e_R(r) - p_R(r) / c ** 2))
        dbetadr *= 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1)

        dHdr = beta
        return [dbetadr, dHdr]

    def tov_eq(self, y, r):
        P, m = y

        if P < self.min_p or P > self.max_p:
            return [0., 0.]

        eden = self.en_dens(P)

        dPdr = -G * (eden + P / c ** 2) * (m + 4.0 * pi * r ** 3 * P / c ** 2) / (r * (r - 2.0 * G * m / c ** 2))

        dmdr = 4.0 * pi * r ** 2 * eden

        return [dPdr, dmdr]

    def solve(self, central_pressure, r_stop=1.0e-8, dr=1e-4):
        r = np.arange(r_stop, 100, dr)
        y0 = [central_pressure, 0]

        sol = odeint(self.tov_eq, y0, r)

        pressure = sol[:, 0]
        mass = sol[:, 1]

        radius = r[pressure > 0][-1]
        mass_final = mass[pressure > 0][-1]

        return radius, mass_final


# Sample energy density and pressure arrays
en_arr = [1e14, 2e14, 3e14, 4e14, 5e14]  # Placeholder values
p_arr = [1e34, 2e34, 3e34, 4e34, 5e34]   # Placeholder values

# Initialize the TOV solver
tov_solver = TOV(en_arr, p_arr)

# Define the range of central densities
central_densities = np.logspace(14, 16, 50)  # Placeholder values in g/cm^3

masses = []
radii = []

# Solve the TOV equations for each central density
for density in central_densities:
    central_pressure = tov_solver.press(density)
    radius, mass = tov_solver.solve(central_pressure)
    masses.append(mass / Msun)  # Convert to solar masses
    radii.append(radius)  # Radius in km

# Plot the mass-radius curve
plt.figure(figsize=(10, 6))
plt.plot(radii, masses, label='Mass-Radius Curve')
plt.xlabel('Radius (km)')
plt.ylabel('Mass (M☉)')
plt.title('Mass-Radius Curve for Neutron Stars')
plt.legend()
plt.grid(True)
plt.show()
import os
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np
import math

# Constants
G = 6.6730831e-8
c = 2.99792458e10
MeV_fm3_to_pa = 1.6021766e35
c_km = 2.99792458e5  # km/s
mN = 1.67e-24  # g
mev_to_ergs = 1.602176565e-6
fm_to_cm = 1.0e-13
ergs_to_mev = 1.0 / mev_to_ergs
cm_to_fm = 1.0 / fm_to_cm
Msun = 1.988435e33
MeV_fm3_to_pa_cgs = 1.6021766e33
km_to_mSun = G / c ** 2
hbarc3 = 197.32700288295746 ** 3
nucleon_mass = 938.04
pi = math.pi


class TOV:
    """
    Instance of the TOV solver
    """

    def __init__(self, en_arr, p_arr, add_crust=True, plot_eos=False):
        print("Initializing TOV solver...")
        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        en_arr *= MeV_fm3_to_pa_cgs / c ** 2
        p_arr *= MeV_fm3_to_pa_cgs
        print("Energy array:", en_arr)
        print("Pressure array:", p_arr)

        sort_ind = np.argsort(p_arr)
        self.en_dens = interp1d(p_arr[sort_ind], en_arr[sort_ind], kind='cubic', bounds_error=False,
                                fill_value=(en_arr[sort_ind][0], en_arr[sort_ind][-1]))

        sort_ind = np.argsort(en_arr)
        self.press = interp1d(en_arr[sort_ind], p_arr[sort_ind], kind='cubic', bounds_error=False,
                              fill_value=(p_arr[sort_ind][0], p_arr[sort_ind][-1]))

        self.__en_arr = en_arr
        self.__p_arr = p_arr

        self.min_dens = np.min(en_arr)
        self.max_dens = np.max(en_arr)

        self.min_p = np.min(p_arr)
        self.max_p = np.max(p_arr)

        if add_crust:
            if plot_eos:
                plt.plot(self.__en_arr / (MeV_fm3_to_pa_cgs / c ** 2),
                         self.__p_arr / MeV_fm3_to_pa_cgs,
                         linestyle='-', label='original EOS')
            self.add_crust()
        if plot_eos:
            plt.plot(self.__en_arr / (MeV_fm3_to_pa_cgs / c ** 2),
                     self.__p_arr / MeV_fm3_to_pa_cgs,
                     linestyle='--', label='EOS with crust')

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'${\rm \varepsilon~(MeV/fm^{3}) }$')
            plt.ylabel(r'${\rm P~(MeV/fm^{3}) }$')
            plt.legend()
            plt.show()

    def add_crust(self):
        """
        Adds Nuclear Statistical Equilibrium crust EOS.
        """
        print("Adding crust...")
        dir_name = os.path.dirname(__file__)
        crust_loc = os.path.join(dir_name, 'data', 'Baym_eos.dat')

        baym_eos = np.genfromtxt(crust_loc, dtype=float, skip_header=1, names=["en", "p", "nB"])

        P_crust = interp1d(baym_eos["en"], baym_eos["p"], kind='cubic', bounds_error=False,
                           fill_value=(baym_eos["p"][0], baym_eos["p"][-1]))

        def eq_glue(n):
            return P_crust(n) - self.press(n)

        g = root(eq_glue, [44. * (MeV_fm3_to_pa_cgs / c ** 2)], options={'maxfev': 200})

        n_glue = g['x'][0]

        en_arr = []
        p_arr = []

        for i in range(len(baym_eos["p"])):
            if baym_eos["en"][i] < n_glue:
                en_arr.append(baym_eos["en"][i])
                p_arr.append(baym_eos["p"][i])
            else:
                break

        for i in range(len(self.__p_arr)):
            if self.__en_arr[i] >= n_glue:
                en_arr.append(self.__en_arr[i])
                p_arr.append(self.__p_arr[i])

        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        self.min_dens = min(en_arr)
        self.min_p = min(p_arr)

        self.en_dens = interp1d(p_arr, en_arr, kind='cubic', bounds_error=False, fill_value=(en_arr[0], en_arr[-1]))
        self.press = interp1d(en_arr, p_arr, kind='cubic', bounds_error=False, fill_value=(p_arr[0], p_arr[-1]))

        self.__en_arr = en_arr
        self.__p_arr = p_arr

    def dedp(self, r, R_dep):
        e_R, p_R, m_R = R_dep

        p = p_R(r)
        dp = p * 0.005

        # Ensure values are within bounds
        p_minus_3dp = max(p - 3 * dp, self.min_p)
        p_minus_2dp = max(p - 2 * dp, self.min_p)
        p_minus_1dp = max(p - 1 * dp, self.min_p)
        p_plus_3dp = min(p + 3 * dp, self.max_p)
        p_plus_2dp = min(p + 2 * dp, self.max_p)
        p_plus_1dp = min(p + 1 * dp, self.max_p)

        el_3 = self.en_dens(p_minus_3dp)
        el_2 = self.en_dens(p_minus_2dp)
        el_1 = self.en_dens(p_minus_1dp)
        er_3 = self.en_dens(p_plus_3dp)
        er_2 = self.en_dens(p_plus_2dp)
        er_1 = self.en_dens(p_plus_1dp)

        de_dp = (-1 / 60 * el_3 + 3 / 20 * el_2 - 3 / 4 * el_1 + 3 / 4 * er_1 - 3 / 20 * er_2 + 1 / 60 * er_3) / dp

        return de_dp

    def love_eq(self, param, r, R_dep):
        beta, H = param
        e_R, p_R, m_R = R_dep

        try:
            dummy = p_R(r)
        except ValueError:
            return [100000, 100000]

        de_dp = self.dedp(r, R_dep)

        dbetadr = H * (-2 * pi * G / c ** 2 * (
                5 * e_R(r) + 9 * p_R(r) / c ** 2 + de_dp * c ** 2 * (e_R(r) + p_R(r) / c ** 2)) \
                       + 3 / r ** 2 \
                       + 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1) * (
                               m_R(r) / r ** 2 * km_to_mSun + G / c ** 4 * 4 * pi * r * p_R(r)) ** 2) \
                  + beta / r * (
                          -1 + m_R(r) / r * km_to_mSun + 2 * pi * r ** 2 * G / c ** 2 * (e_R(r) - p_R(r) / c ** 2))
        dbetadr *= 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1)

        dHdr = beta
        return [dbetadr, dHdr]

    def tov_eq(self, y, r):
        P, m = y

        if P < self.min_p or P > self.max_p:
            return [0., 0.]

        eden = self.en_dens(P)

        dPdr = -G * (eden + P / c ** 2) * (m + 4.0 * pi * r ** 3 * P / c ** 2) / (r * (r - 2.0 * G * m / c ** 2))

        dmdr = 4.0 * pi * r ** 2 * eden

        return [dPdr, dmdr]

    def solve(self, central_pressure, r_stop=1.0e-8, dr=1e-4):
        r = np.arange(r_stop, 100, dr)
        y0 = [central_pressure, 0]

        sol = odeint(self.tov_eq, y0, r)

        pressure = sol[:, 0]
        mass = sol[:, 1]

        radius = r[pressure > 0][-1]
        mass_final = mass[pressure > 0][-1]

        return radius, mass_final


# Sample energy density and pressure arrays
en_arr = [1e14, 2e14, 3e14, 4e14, 5e14]  # Placeholder values
p_arr = [1e34, 2e34, 3e34, 4e34, 5e34]   # Placeholder values

# Initialize the TOV solver
tov_solver = TOV(en_arr, p_arr)

# Define the range of central densities
central_densities = np.logspace(14, 16, 50)  # Placeholder values in g/cm^3

masses = []
radii = []

# Solve the TOV equations for each central density
for density in central_densities:
    central_pressure = tov_solver.press(density)
    radius, mass = tov_solver.solve(central_pressure)
    masses.append(mass / Msun)  # Convert to solar masses
    radii.append(radius)  # Radius in km

# Plot the mass-radius curve
plt.figure(figsize=(10, 6))
plt.plot(radii, masses, label='Mass-Radius Curve')
plt.xlabel('Radius (km)')
plt.ylabel('Mass (M☉)')
plt.title('Mass-Radius Curve for Neutron Stars')
plt.legend()
plt.grid(True)
plt.show()
import os
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np
import math

# Constants
G = 6.6730831e-8
c = 2.99792458e10
MeV_fm3_to_pa = 1.6021766e35
c_km = 2.99792458e5  # km/s
mN = 1.67e-24  # g
mev_to_ergs = 1.602176565e-6
fm_to_cm = 1.0e-13
ergs_to_mev = 1.0 / mev_to_ergs
cm_to_fm = 1.0 / fm_to_cm
Msun = 1.988435e33
MeV_fm3_to_pa_cgs = 1.6021766e33
km_to_mSun = G / c ** 2
hbarc3 = 197.32700288295746 ** 3
nucleon_mass = 938.04
pi = math.pi


class TOV:
    """
    Instance of the TOV solver
    """

    def __init__(self, en_arr, p_arr, add_crust=True, plot_eos=False):
        print("Initializing TOV solver...")
        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        en_arr *= MeV_fm3_to_pa_cgs / c ** 2
        p_arr *= MeV_fm3_to_pa_cgs
        print("Energy array:", en_arr)
        print("Pressure array:", p_arr)

        sort_ind = np.argsort(p_arr)
        self.en_dens = interp1d(p_arr[sort_ind], en_arr[sort_ind], kind='cubic', bounds_error=False,
                                fill_value=(en_arr[sort_ind][0], en_arr[sort_ind][-1]))

        sort_ind = np.argsort(en_arr)
        self.press = interp1d(en_arr[sort_ind], p_arr[sort_ind], kind='cubic', bounds_error=False,
                              fill_value=(p_arr[sort_ind][0], p_arr[sort_ind][-1]))

        self.__en_arr = en_arr
        self.__p_arr = p_arr

        self.min_dens = np.min(en_arr)
        self.max_dens = np.max(en_arr)

        self.min_p = np.min(p_arr)
        self.max_p = np.max(p_arr)

        if add_crust:
            if plot_eos:
                plt.plot(self.__en_arr / (MeV_fm3_to_pa_cgs / c ** 2),
                         self.__p_arr / MeV_fm3_to_pa_cgs,
                         linestyle='-', label='original EOS')
            self.add_crust()
        if plot_eos:
            plt.plot(self.__en_arr / (MeV_fm3_to_pa_cgs / c ** 2),
                     self.__p_arr / MeV_fm3_to_pa_cgs,
                     linestyle='--', label='EOS with crust')

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'${\rm \varepsilon~(MeV/fm^{3}) }$')
            plt.ylabel(r'${\rm P~(MeV/fm^{3}) }$')
            plt.legend()
            plt.show()

    def add_crust(self):
        """
        Adds Nuclear Statistical Equilibrium crust EOS.
        """
        print("Adding crust...")
        dir_name = os.path.dirname(__file__)
        crust_loc = os.path.join(dir_name, 'data', 'Baym_eos.dat')

        baym_eos = np.genfromtxt(crust_loc, dtype=float, skip_header=1, names=["en", "p", "nB"])

        P_crust = interp1d(baym_eos["en"], baym_eos["p"], kind='cubic', bounds_error=False,
                           fill_value=(baym_eos["p"][0], baym_eos["p"][-1]))

        def eq_glue(n):
            return P_crust(n) - self.press(n)

        g = root(eq_glue, [44. * (MeV_fm3_to_pa_cgs / c ** 2)], options={'maxfev': 200})

        n_glue = g['x'][0]

        en_arr = []
        p_arr = []

        for i in range(len(baym_eos["p"])):
            if baym_eos["en"][i] < n_glue:
                en_arr.append(baym_eos["en"][i])
                p_arr.append(baym_eos["p"][i])
            else:
                break

        for i in range(len(self.__p_arr)):
            if self.__en_arr[i] >= n_glue:
                en_arr.append(self.__en_arr[i])
                p_arr.append(self.__p_arr[i])

        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        self.min_dens = min(en_arr)
        self.min_p = min(p_arr)

        self.en_dens = interp1d(p_arr, en_arr, kind='cubic', bounds_error=False, fill_value=(en_arr[0], en_arr[-1]))
        self.press = interp1d(en_arr, p_arr, kind='cubic', bounds_error=False, fill_value=(p_arr[0], p_arr[-1]))

        self.__en_arr = en_arr
        self.__p_arr = p_arr

    def dedp(self, r, R_dep):
        e_R, p_R, m_R = R_dep

        p = p_R(r)
        dp = p * 0.005

        # Ensure values are within bounds
        p_minus_3dp = max(p - 3 * dp, self.min_p)
        p_minus_2dp = max(p - 2 * dp, self.min_p)
        p_minus_1dp = max(p - 1 * dp, self.min_p)
        p_plus_3dp = min(p + 3 * dp, self.max_p)
        p_plus_2dp = min(p + 2 * dp, self.max_p)
        p_plus_1dp = min(p + 1 * dp, self.max_p)

        el_3 = self.en_dens(p_minus_3dp)
        el_2 = self.en_dens(p_minus_2dp)
        el_1 = self.en_dens(p_minus_1dp)
        er_3 = self.en_dens(p_plus_3dp)
        er_2 = self.en_dens(p_plus_2dp)
        er_1 = self.en_dens(p_plus_1dp)

        de_dp = (-1 / 60 * el_3 + 3 / 20 * el_2 - 3 / 4 * el_1 + 3 / 4 * er_1 - 3 / 20 * er_2 + 1 / 60 * er_3) / dp

        return de_dp

    def love_eq(self, param, r, R_dep):
        beta, H = param
        e_R, p_R, m_R = R_dep

        try:
            dummy = p_R(r)
        except ValueError:
            return [100000, 100000]

        de_dp = self.dedp(r, R_dep)

        dbetadr = H * (-2 * pi * G / c ** 2 * (
                5 * e_R(r) + 9 * p_R(r) / c ** 2 + de_dp * c ** 2 * (e_R(r) + p_R(r) / c ** 2)) \
                       + 3 / r ** 2 \
                       + 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1) * (
                               m_R(r) / r ** 2 * km_to_mSun + G / c ** 4 * 4 * pi * r * p_R(r)) ** 2) \
                  + beta / r * (
                          -1 + m_R(r) / r * km_to_mSun + 2 * pi * r ** 2 * G / c ** 2 * (e_R(r) - p_R(r) / c ** 2))
        dbetadr *= 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1)

        dHdr = beta
        return [dbetadr, dHdr]

    def tov_eq(self, y, r):
        P, m = y

        if P < self.min_p or P > self.max_p:
            return [0., 0.]

        eden = self.en_dens(P)

        dPdr = -G * (eden + P / c ** 2) * (m + 4.0 * pi * r ** 3 * P / c ** 2) / (r * (r - 2.0 * G * m / c ** 2))

        dmdr = 4.0 * pi * r ** 2 * eden

        return [dPdr, dmdr]

    def solve(self, central_pressure, r_stop=1.0e-8, dr=1e-4):
        r = np.arange(r_stop, 100, dr)
        y0 = [central_pressure, 0]

        sol = odeint(self.tov_eq, y0, r)

        pressure = sol[:, 0]
        mass = sol[:, 1]

        radius = r[pressure > 0][-1]
        mass_final = mass[pressure > 0][-1]

        return radius, mass_final


# Sample energy density and pressure arrays
en_arr = [1e14, 2e14, 3e14, 4e14, 5e14]  # Placeholder values
p_arr = [1e34, 2e34, 3e34, 4e34, 5e34]   # Placeholder values

# Initialize the TOV solver
tov_solver = TOV(en_arr, p_arr)

# Define the range of central densities
central_densities = np.logspace(14, 16, 50)  # Placeholder values in g/cm^3

masses = []
radii = []

# Solve the TOV equations for each central density
for density in central_densities:
    central_pressure = tov_solver.press(density)
    radius, mass = tov_solver.solve(central_pressure)
    masses.append(mass / Msun)  # Convert to solar masses
    radii.append(radius)  # Radius in km

# Plot the mass-radius curve
plt.figure(figsize=(10, 6))
plt.plot(radii, masses, label='Mass-Radius Curve')
plt.xlabel('Radius (km)')
plt.ylabel('Mass (M☉)')
plt.title('Mass-Radius Curve for Neutron Stars')
plt.legend()
plt.grid(True)
plt.show()
