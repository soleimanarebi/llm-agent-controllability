"""Core model for controllability of LLM agent networks.

Each agent holds a scalar state x_i (one coordinate of its answer state, e.g. the
log-odds it assigns to the verified answer s). One round of reading and rewriting is

    x_i(t+1) = x_i + alpha*(x_i - s) + c * sum_j a_ij * phi(x_j - x_i) + u_i(t)

with phi = tanh (bounded influence) or identity (linear model). In error
coordinates e = x - s the linear model is e(t+1) = A e(t) + B u(t), with
A = (1 + alpha) I - c L.
"""
import numpy as np
import networkx as nx
import scipy.linalg as sla


# ---------------------------------------------------------------- graphs

def make_graph(kind, n=64, seed=0):
    rng = np.random.default_rng(seed)
    s = int(rng.integers(1 << 30))
    if kind == "complete":
        g = nx.complete_graph(n)
    elif kind == "star":
        g = nx.star_graph(n - 1)
    elif kind == "ring":
        g = nx.watts_strogatz_graph(n, 6, 0.0, seed=s)
    elif kind == "small-world":
        g = nx.connected_watts_strogatz_graph(n, 6, 0.1, seed=s)
    elif kind == "random":
        while True:
            g = nx.gnm_random_graph(n, 3 * n, seed=s)
            if nx.is_connected(g):
                break
            s += 1
    elif kind == "scale-free":
        g = nx.barabasi_albert_graph(n, 3, seed=s)
    else:
        raise ValueError(kind)
    return nx.convert_node_labels_to_integers(g)


def laplacian(g):
    a = nx.to_numpy_array(g, nodelist=range(g.number_of_nodes()))
    return np.diag(a.sum(1)) - a, a


# ------------------------------------------------- structural controllability

def driver_count_no_memory(dg):
    """Liu-Slotine-Barabasi minimum driver nodes for a digraph without self-loops.

    Edge (j, i) means agent i reads agent j (j influences i). N_D = max(N - |M*|, 1)
    with M* a maximum matching of the bipartite out/in representation.
    """
    n = dg.number_of_nodes()
    b = nx.Graph()
    b.add_nodes_from((("o", v) for v in dg.nodes), bipartite=0)
    b.add_nodes_from((("i", v) for v in dg.nodes), bipartite=1)
    b.add_edges_from((("o", u), ("i", v)) for u, v in dg.edges if u != v)
    top = [("o", v) for v in dg.nodes]
    m = nx.bipartite.hopcroft_karp_matching(b, top_nodes=top)
    size = sum(1 for k in m if k[0] == "o")
    return max(n - size, 1)


def driver_count_with_memory(dg):
    """Minimum drivers when every agent also carries its own previous state
    (a self-loop on every node): one driver per source strongly connected component."""
    cond = nx.condensation(dg)
    return sum(1 for v in cond.nodes if cond.in_degree(v) == 0)


def directed_versions(n=64, seed=0):
    """Directed influence graphs typical of agent harnesses. Edge u->v: v reads u."""
    out = {}
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from((0, i) for i in range(1, n))          # orchestrator broadcasts
    out["broadcast star"] = g.copy()
    g.add_edges_from((i, 0) for i in range(1, n))          # subagents report back
    out["orchestrator star"] = g
    t = nx.balanced_tree(3, 3)                              # 40 nodes
    dt = nx.bfs_tree(t, 0)
    out["delegation tree"] = dt
    p = nx.DiGraph()
    p.add_nodes_from(range(n))
    width = 8                                               # 8 parallel 8-stage pipelines
    for k in range(width):
        nodes = [k * 8 + j for j in range(8)]
        p.add_edges_from(zip(nodes[:-1], nodes[1:]))
    out["parallel pipelines"] = p
    for kind in ["ring", "small-world", "random", "scale-free"]:
        out[kind + " (debate)"] = make_graph(kind, n, seed).to_directed()
    return out


# ------------------------------------------------------------ control energy

def gramian(A, B, T):
    W = np.zeros((A.shape[0], A.shape[0]))
    M = B.copy()
    for _ in range(T):
        W += M @ M.T
        M = A @ M
    return W


def energy_stats(A, B, T):
    """Return lambda_min(W_T) and average reachability energy tr(W_T^-1)/N."""
    W = gramian(A, B, T)
    ev = np.linalg.eigvalsh((W + W.T) / 2)
    ev = np.clip(ev, 1e-300, None)
    return ev[0], np.sum(1.0 / ev) / len(ev)


# -------------------------------------------------------- pinned-set rules

def pick_pinned(g, L, k, rule, c=0.05, gain=0.5, seed=0):
    n = L.shape[0]
    if rule == "random":
        return list(np.random.default_rng(seed).choice(n, k, replace=False))
    if rule == "degree":
        return [v for v, _ in sorted(g.degree, key=lambda t: (-t[1], t[0]))[:k]]
    if rule == "betweenness":
        b = nx.betweenness_centrality(g)
        return sorted(b, key=lambda v: (-b[v], v))[:k]
    if rule == "spectral-greedy":
        chosen = []
        for _ in range(k):
            best, bv = None, -np.inf
            for v in range(n):
                if v in chosen:
                    continue
                p = np.zeros(n)
                p[chosen + [v]] = 1
                mu1 = np.linalg.eigvalsh(c * L + gain * np.diag(p))[0]
                if mu1 > bv:
                    best, bv = v, mu1
            chosen.append(best)
        return chosen
    raise ValueError(rule)


# ---------------------------------------------------------- simulation

def step(e, Aadj, alpha, c, phi):
    diff = e[None, :] - e[:, None]                      # diff[i, j] = e_j - e_i
    infl = (Aadj * phi(diff)).sum(1)
    return e + alpha * e + c * infl


def run_proportional(e0, Aadj, alpha, c, gain, pinned, phi=np.tanh, T=2000,
                     tol=0.05, offsets=None, umax=None):
    """Proportional pinning u_i = -g (e_i - offset_i). Returns (rounds, energy, e_final)."""
    n = len(e0)
    p = np.zeros(n)
    p[pinned] = 1
    off = np.zeros(n) if offsets is None else offsets
    e = e0.copy()
    scale = np.max(np.abs(e0))
    energy = 0.0
    for t in range(T):
        u = -gain * p * (e - off)
        if umax is not None:
            u = np.clip(u, -umax, umax)
        energy += float(u @ u)
        e = step(e, Aadj, alpha, c, phi) + u
        if not np.all(np.isfinite(e)) or np.max(np.abs(e)) > 1e6:
            return None, energy, e
        if offsets is None and np.max(np.abs(e)) < tol * scale:
            return t + 1, energy, e
    return (None if offsets is None else T), energy, e


def sdc_laplacian(e, Aadj):
    """State-dependent coefficient form of tanh coupling: tanh(z) = (tanh(z)/z) z."""
    z = e[None, :] - e[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(z) < 1e-8, 1.0, np.tanh(z) / z)
    W = Aadj * w
    return np.diag(W.sum(1)) - W


def run_sdre(e0, Aadj, alpha, c, pinned, q=1.0, r=1.0, T=2000, tol=0.05, umax=None):
    """Discrete-time SDRE pinning: at each round solve the DARE for (A(e), B)."""
    n = len(e0)
    B = np.zeros((n, len(pinned)))
    B[pinned, range(len(pinned))] = 1
    Q = q * np.eye(n)
    R = r * np.eye(len(pinned))
    e = e0.copy()
    scale = np.max(np.abs(e0))
    energy = 0.0
    for t in range(T):
        A = (1 + alpha) * np.eye(n) - c * sdc_laplacian(e, Aadj)
        X = sla.solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A)
        u = -K @ e
        if umax is not None:
            u = np.clip(u, -umax, umax)
        energy += float(u @ u)
        e = step(e, Aadj, alpha, c, np.tanh) + B @ u
        if not np.all(np.isfinite(e)) or np.max(np.abs(e)) > 1e6:
            return None, energy, e
        if np.max(np.abs(e)) < tol * scale:
            return t + 1, energy, e
    return None, energy, e


# ------------------------------------------------------- theory helpers

def mu_extremes(L, c, gain, pinned, w=1.0):
    n = L.shape[0]
    p = np.zeros(n)
    p[pinned] = 1
    ev = np.linalg.eigvalsh(c * w * L + gain * np.diag(p))
    return ev[0], ev[-1]


def harmonic_capture(L, c, gain, pinned, compromised):
    """h = (cL + gP)^{-1} g P_c 1: steady-state fraction of the hijack offset per agent."""
    n = L.shape[0]
    p = np.zeros(n)
    p[pinned] = 1
    pc = np.zeros(n)
    pc[compromised] = 1
    return np.linalg.solve(c * L + gain * np.diag(p), gain * pc)


# ------------------------------------------------------ exact controllability

def influence_matrix(dg):
    """a[i, j] = 1 if agent i reads agent j (edge j -> i)."""
    n = dg.number_of_nodes()
    return nx.to_numpy_array(dg, nodelist=range(n)).T


def exact_driver_count(A, tol=1e-8):
    """Yuan et al. (2013): minimum drivers for exact controllability of (A, B)
    equals the maximum geometric multiplicity max_lambda [N - rank(lambda I - A)]."""
    n = A.shape[0]
    ev = np.linalg.eigvals(A)
    best = 1
    seen = []
    for lam in ev:
        if any(abs(lam - s) < 1e-6 for s in seen):
            continue
        seen.append(lam)
        sv = np.linalg.svd(lam * np.eye(n) - A, compute_uv=False)
        best = max(best, int(np.sum(sv < tol * max(1.0, sv[0]))))
    return best


def uncontrolled_modes(A_sym, drivers, tol=1e-9):
    """PBH test for symmetric A: number of modes not reached by the driver set."""
    ev, V = np.linalg.eigh(A_sym)
    n = len(ev)
    count, i = 0, 0
    while i < n:
        j = i
        while j + 1 < n and abs(ev[j + 1] - ev[i]) < 1e-8:
            j += 1
        Vs = V[:, i:j + 1][drivers, :]
        r = np.linalg.matrix_rank(Vs, tol=tol) if Vs.size else 0
        count += (j - i + 1) - r
        i = j + 1
    return count


# ------------------------------------- performance-based pinning controllability

def centrality_order(g, rule, seed=0):
    """Agents ranked for pinning by a centrality criterion (highest first)."""
    n = g.number_of_nodes()
    if rule == "betweenness":
        s = nx.betweenness_centrality(g)
    elif rule == "closeness":
        s = nx.closeness_centrality(g)
    elif rule == "eigenvector":
        s = nx.eigenvector_centrality_numpy(g)
    elif rule == "degree":
        s = dict(g.degree)
    elif rule == "random":
        perm = np.random.default_rng(seed).permutation(n)
        return [int(v) for v in perm]
    else:
        raise ValueError(rule)
    return sorted(range(n), key=lambda v: (-s[v], v))


def contraction(L, c, gain, alpha, pinned):
    """rho = max_i |1 + alpha - mu_i(cL + gP)|: per-round worst-case contraction (Theorem 1)."""
    mu1, muN = mu_extremes(L, c, gain, pinned)
    return max(abs(1 + alpha - mu1), abs(1 + alpha - muN))


def min_drivers_linear(L, order, c, gain, alpha, rho_target):
    """Smallest k such that pinning the first k agents of `order` gives rho <= rho_target."""
    n = L.shape[0]
    for k in range(1, n + 1):
        if contraction(L, c, gain, alpha, order[:k]) <= rho_target:
            return k
    return None


def greedy_order(L, c, gain, alpha, kmax=None):
    """Spectral-greedy ranking: add the agent that most reduces rho (benchmark)."""
    n = L.shape[0]
    kmax = n if kmax is None else kmax
    chosen = []
    for _ in range(kmax):
        best, bv = None, np.inf
        for v in range(n):
            if v in chosen:
                continue
            r = contraction(L, c, gain, alpha, chosen + [v])
            if r < bv:
                best, bv = v, r
        chosen.append(best)
    return chosen


def settling_rounds(e0, Aadj, alpha, c, gain, pinned, phi=np.tanh, T=200, eps=0.05):
    """Rounds until ||e(t)||_2 <= eps ||e(0)||_2 under proportional pinning (None if not within T)."""
    n = len(e0)
    p = np.zeros(n)
    p[pinned] = 1
    e = e0.copy()
    n0 = np.linalg.norm(e0)
    for t in range(T):
        e = step(e, Aadj, alpha, c, phi) - gain * p * e
        if not np.all(np.isfinite(e)):
            return None
        if np.linalg.norm(e) <= eps * n0:
            return t + 1
    return None


def make_graph_density(kind, n, dmean, seed=0):
    """ER, small-world or scale-free graph with mean degree about dmean, connected."""
    s = int(np.random.default_rng(seed).integers(1 << 30))
    for _ in range(100):
        if kind == "random":
            g = nx.gnm_random_graph(n, n * dmean // 2, seed=s)
        elif kind == "small-world":
            g = nx.watts_strogatz_graph(n, dmean, 0.1, seed=s)
        elif kind == "scale-free":
            g = nx.barabasi_albert_graph(n, dmean // 2, seed=s)
        else:
            raise ValueError(kind)
        if nx.is_connected(g):
            return nx.convert_node_labels_to_integers(g)
        s += 1
    raise RuntimeError("no connected graph")


def settling_controlled(e0, Aadj, alpha, c, pinned, ctrl, gain=0.5, q=1.0, r=1.0,
                        umax=1.0, T=40, eps=0.05):
    """Rounds until ||e||_2 <= eps ||e0||_2 under bounded influence, with proportional
    ('P') or discrete SDRE ('SDRE') pinning saturated at umax. Returns (rounds, energy)."""
    n = len(e0)
    k = len(pinned)
    B = np.zeros((n, k))
    B[pinned, range(k)] = 1
    Q, R = q * np.eye(n), r * np.eye(k)
    e = e0.copy()
    n0 = np.linalg.norm(e0)
    energy = 0.0
    for t in range(T):
        if ctrl == "P":
            v = -gain * e[pinned]
        else:
            A = (1 + alpha) * np.eye(n) - c * sdc_laplacian(e, Aadj)
            X = sla.solve_discrete_are(A, B, Q, R)
            v = -np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A) @ e
        v = np.clip(v, -umax, umax)
        energy += float(v @ v)
        e = step(e, Aadj, alpha, c, np.tanh) + B @ v
        if not np.all(np.isfinite(e)):
            return None, energy
        if np.linalg.norm(e) <= eps * n0:
            return t + 1, energy
    return None, energy
