from types import SimpleNamespace
import numpy as np

pauli = SimpleNamespace(
    s0=np.array([[1.0, 0.0], [0.0, 1.0]]),
    sx=np.array([[0.0, 1.0], [1.0, 0.0]]),
    sy=np.array([[0.0, -1j], [1j, 0.0]]),
    sz=np.array([[1.0, 0.0], [0.0, -1.0]]),
)

pauli.s0s0 = np.kron(pauli.s0, pauli.s0)
pauli.s0sx = np.kron(pauli.s0, pauli.sx)
pauli.s0sy = np.kron(pauli.s0, pauli.sy)
pauli.s0sz = np.kron(pauli.s0, pauli.sz)
pauli.sxs0 = np.kron(pauli.sx, pauli.s0)
pauli.sxsx = np.kron(pauli.sx, pauli.sx)
pauli.sxsy = np.kron(pauli.sx, pauli.sy)
pauli.sxsz = np.kron(pauli.sx, pauli.sz)
pauli.sys0 = np.kron(pauli.sy, pauli.s0)
pauli.sysx = np.kron(pauli.sy, pauli.sx)
pauli.sysy = np.kron(pauli.sy, pauli.sy)
pauli.sysz = np.kron(pauli.sy, pauli.sz)
pauli.szs0 = np.kron(pauli.sz, pauli.s0)
pauli.szsx = np.kron(pauli.sz, pauli.sx)
pauli.szsy = np.kron(pauli.sz, pauli.sy)
pauli.szsz = np.kron(pauli.sz, pauli.sz)
