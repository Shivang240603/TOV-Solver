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

    def _init_(self, en_arr, p_arr, add_crust=True, plot_eos=False):
        print("Initializing TOV solver...")
        en_arr = np.array(en_arr, dtype=float)
        p_arr = np.array(p_arr, dtype=float)

        en_arr = MeV_fm3_to_pa_cgs / c * 2
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
        dir_name = os.path.dirname(_file_)
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
                5 * e_R(r) + 9 * p_R(r) / c * 2 + de_dp * c * 2 * (e_R(r) + p_R(r) / c ** 2)) \
                       + 3 / r ** 2 \
                       + 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1) * (
                               m_R(r) / r * 2 * km_to_mSun + G / c * 4 * 4 * pi * r * p_R(r)) ** 2) \
                  + beta / r * (
                          -1 + m_R(r) / r * km_to_mSun + 2 * pi * r * 2 * G / c * 2 * (e_R(r) - p_R(r) / c ** 2))
        dbetadr = 2 * (1 - 2 * m_R(r) / r * km_to_mSun) * (-1)

        dHdr = beta
        return [dbetadr, dHdr]

    def tov_eq(self, y, r):
        P, m = y

        if P < self.min_p or P > self.max_p:
            return [0., 0.]

        eden = self.en_dens(P)

        dPdr = -G * (eden + P / c * 2) * (m + 4.0 * pi * r * 3 * P / c * 2) / (r * (r - 2.0 * G * m / c * 2))

        dmdr = 4.0 * pi * r ** 2 * eden

        return [dPdr, dmdr]

    def check_density(self, central_density):
        if central_density < self.min_dens or central_density > self.max_dens:
            raise ValueError(f"Central density is out of range. Must be between {self.min_dens} and {self.max_dens}")

    def max_radius(self, central_density, p):
        max_r = 100
        tmp_p = np.logspace(np.log10(p), np.log10(self.max_p), 100)
        rho_c = self.en_dens(central_density)

        for i in range(len(tmp_p)):
            if self.en_dens(tmp_p[i]) > rho_c:
                max_r = max_r * tmp_p[i] / self.max_p
                break
        return max_r

    def solve(self, central_density, R_stop=None, dr=1e-3, max_dr=None, solver="LSODA", verbose=True):
        if verbose:
            print("Solving TOV equations...")

        self.check_density(central_density)

        r_0 = 1e-6
        R_max = 15.0

        if R_stop is not None:
            R_max = R_stop

        P_0 = self.press(central_density)

        if verbose:
            print("Initial pressure P_0:", P_0)

        init_cond = [P_0, 4 * pi / 3.0 * r_0 ** 3 * central_density]

        r = np.arange(r_0, R_max, dr, dtype=float)
        sol = odeint(self.tov_eq, init_cond, r, mxstep=5000)

        r_surf = 0.0
        index_surf = 0
        for j in range(len(sol[:, 0])):
            if sol[j, 0] < self.min_p:
                r_surf = r[j]
                index_surf = j
                break
        if verbose:
            print("Surface radius:", r_surf)
        if r_surf == 0.0:
            r_surf = r[-1]
            index_surf = -1

        R_dep = (interp1d(r[:index_surf], self.en_dens(sol[:index_surf, 0]), kind='linear', bounds_error=False,
                          fill_value="extrapolate"),
                 interp1d(r[:index_surf], sol[:index_surf, 0], kind='linear', bounds_error=False,
                          fill_value="extrapolate"),
                 interp1d(r[:index_surf], sol[:index_surf, 1], kind='linear', bounds_error=False,
                          fill_value="extrapolate"))

        sol_love = odeint(self.love_eq, [2., 2.], r[:index_surf], args=(R_dep,))

        yR = sol_love[-1, 1]

        C = G * sol[index_surf, 1] / (r_surf * c ** 2)

        k2 = (8.0 / 5.0 * C * 5 * (1 - 2 * C) * 2 * (2 + 2 * C * (yR - 1) - yR) / (
                2 * C * (6 - 3 * yR + 3 * C * (5 * yR - 8)) + 4 * C ** 3 * (
                    13 - 11 * yR + C * (3 * yR - 2) + 2 * C ** 2 * (1 + yR)) +
                3 * (1 - 2 * C) ** 2 * (2 - yR + 2 * C * (yR - 1)) * math.log(1 - 2 * C)))

        lam = 2.0 / (3.0 * k2) * (r_surf * c * 2 / G) * 5
        Love = lam / (sol[index_surf, 1] * r_surf * 2) * 5

        if verbose:
            print("Final results:")
            print("Central density (in g/cm^3):", central_density * c ** 2 / MeV_fm3_to_pa_cgs)
            print("Mass (in solar masses):", sol[index_surf, 1] / Msun)
            print("Radius (in km):", r_surf)
            print("Love number:", Love)
            print("Compactness:", C)

        return {'r': r[:index_surf], 'P': sol[:index_surf, 0], 'm': sol[:index_surf, 1], 'Love': Love, 'R_surf': r_surf,
                'k2': k2}


# Sample usage
# Define energy and pressure arrays (replace these with actual data)
en_arr = [100, 200, 300, 400, 500]
p_arr = [10, 20, 30, 40, 50]

# Create an instance of TOV
tov_solver = TOV(en_arr, p_arr, add_crust=True, plot_eos=True)

# Solve for a given central density (replace 1e15 with actual value)
result = tov_solver.solve(central_density=1e15, verbose=True)
print(result)
