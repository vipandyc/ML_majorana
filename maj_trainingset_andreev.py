import kwant
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tbmodel_setting import pauli

def onsite_barrier(site, t, mu_n, V_barrier, Ez):
    return (2 * t - mu_n + V_barrier) * pauli.s0sz + Ez * pauli.sxs0
def onsite_normal(site, t, mu_n, Ez):
    return (2 * t - mu_n) * pauli.s0sz + Ez * pauli.sxs0
def onsite_sc(site, t, mu_sc, Ez, delta):
    return (2 * t - mu_sc) * pauli.s0sz + Ez * pauli.sxs0 + delta * pauli.s0sx
def hop(site1, site2, t, alpha):
    return -t * pauli.s0sz + 0.5j * alpha * pauli.sysz

# bad & ugly potentials
def onsite_sc_bad(site, t, mu_sc, Ez, V_max, delta):
    x = site.pos[0] * 10
    sigma = 0.15 * 1000
    return (2 * t - mu_sc + V_max * np.exp(-x**2/2/sigma**2)) * pauli.s0sz + Ez * pauli.sxs0 + delta * pauli.s0sx

def onsite_sc_ugly(site, t, mu_sc, Ez, randlst, delta):
    m = randlst[int(site.pos[0]) - 1]
    return (2 * t - mu_sc + m) * pauli.s0sz + Ez * pauli.sxs0 + delta * pauli.s0sx

def make_wire(onsite_sc_potential, N): # L = N * 10nm
    '''
    onsite_sc_potential is a function for superconducting onsite energy
    '''
    lat = kwant.lattice.chain(norbs=4)

    # The nanowire SM+SC
    wire = kwant.Builder()
    wire[lat(0)] = onsite_barrier
    for i in range(N):
        wire[lat(i + 1)] = onsite_sc_potential
        wire[lat(i), lat(i + 1)] = hop

    # Normal lead: charge conservation and no gap
    normal_lead = kwant.Builder(
        kwant.TranslationalSymmetry((-1, )),
        conservation_law=-pauli.s0sz,
    )

    # Length = 1 only one normal_lead
    normal_lead[lat(0)] = onsite_normal

    # Superconductor has a gap.
    superconductor = kwant.Builder(kwant.TranslationalSymmetry((1, )))
    superconductor[lat(0)] = onsite_sc_potential

    normal_lead[lat(1), lat(0)] = superconductor[lat(1), lat(0)] = hop

    # Attach leads to the scattering region
    wire.attach_lead(normal_lead)
    wire.attach_lead(superconductor)
    wire = wire.finalized()
    return wire

def Andreev_conductance(syst, params, energy):
    smatrix = kwant.smatrix(syst, energy=energy, params=params, in_leads=[0], out_leads=[0])
    n_modes = smatrix.lead_info[0].block_nmodes[0]
    return (
        n_modes
        - smatrix.transmission((0, 0), (0, 0))
        + smatrix.transmission((0, 1), (0, 0))
    )

def calc_spectroscopy(wire_syst, Ez, energy):
    # Trivial, because the magnetic field is zero (third argument)
    params["Ez"] = Ez
    didv = Andreev_conductance(wire_syst, params, energy)

    return didv

def draw_diagram(wire_syst, N, chemical_disorder=0):
    Ez_lst = np.linspace(0, 2.24, 28)
    energy_lst = np.linspace(-0.28, 0.28, 28)

    EZ, ENERGY = np.meshgrid(Ez_lst, energy_lst)
    DIDV = np.zeros(np.shape(EZ))

    # a fixed randomlst for sweeping ONE Hamiltonian
    params["randlst"] = chemical_disorder * np.random.normal(size=N)
    for i in range(len(DIDV)):
        for j in range(len(DIDV[0])):
            res = calc_spectroscopy(wire_syst, Ez_lst[j], energy_lst[i])
            DIDV[i][j] = res

    return DIDV


Nsamples = 2000
# Generate samples
Bad_DIDV = np.zeros((Nsamples, 28, 28))

for i in range(Nsamples):
    params = dict(
        t=25, mu_n=25, mu_sc=1, alpha=5, delta=0.2, V_barrier=10, V_max=1.5
    )
    params["mu_sc"] = 1 + 0.5 * np.random.normal()
    params["V_barrier"] = 10 + 5 * np.random.normal()
    params["alpha"] = 5 + 2.5 * np.random.normal()
    params["V_max"] = 1.5 + 0.75 * np.random.normal()
    params["delta"] = 0.2 + 0.1 * np.random.normal()
    N0 = np.random.randint(100, 300)

    wire = make_wire(onsite_sc_bad, N=N0)
    Bad_DIDV[i] = draw_diagram(wire, N=N0, chemical_disorder=0.25)

np.save('Andreev_majsts_data.npy', Bad_DIDV)
