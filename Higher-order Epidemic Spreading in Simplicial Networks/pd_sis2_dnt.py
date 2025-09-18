import random
import numpy as np
import networkx as nx
from multiprocessing import Pool

class SimplagionModel:
    def __init__(self, neighbors_dict, triangles_list):
        self.neighbors_dict = neighbors_dict
        self.triangles_list = [tuple(sorted(t)) for t in triangles_list if len(set(t)) == 3]
        self.nodes = list(neighbors_dict.keys())
        self.N = len(self.nodes)
        self.iAgentSet = set()
        self.sAgentSet = set()
        self.rAgentSet = set()
        self.iList = []
        self.t = 0

    def initial_setup(self, infected, immune):
        self.sAgentSet = set(self.nodes)
        self.iAgentSet = set(infected) & self.sAgentSet
        self.sAgentSet -= self.iAgentSet
        self.rAgentSet = set(immune) & self.sAgentSet
        self.sAgentSet -= self.rAgentSet
        self.iList = []
        self.t = 0
        return list(self.iAgentSet), list(self.rAgentSet)

    def infectAgent(self, agent):
        if agent in self.sAgentSet:
            self.iAgentSet.add(agent)
            self.sAgentSet.remove(agent)
            return 1
        return 0

    def recoverAgent(self, agent):
        if agent in self.iAgentSet:
            self.sAgentSet.add(agent)
            self.iAgentSet.remove(agent)
            return 1
        return 0

    def immAgent(self, agent):
        if agent in self.sAgentSet:
            self.rAgentSet.add(agent)
            self.sAgentSet.remove(agent)
            return 1
        return 0

    def run(self, t_max, beta1, beta2, mu, linjie):
        self.t_max = t_max
        while self.iAgentSet and self.sAgentSet and self.t <= self.t_max:
            newI, newR1 = set(), set()
            for iAgent in self.iAgentSet:
                for agent in self.neighbors_dict.get(iAgent, ()):
                    if agent in self.sAgentSet and random.random() <= beta1:
                        newI.add(agent)
            for n1, n2, n3 in self.triangles_list:
                if n1 in self.iAgentSet and n2 in self.iAgentSet and n3 in self.sAgentSet and random.random() <= 2 * beta2:
                    newI.add(n3)
                elif n1 in self.iAgentSet and n3 in self.iAgentSet and n2 in self.sAgentSet and random.random() <= 2 * beta2:
                    newI.add(n2)
                elif n2 in self.iAgentSet and n3 in self.iAgentSet and n1 in self.sAgentSet and random.random() <= 2 * beta2:
                    newI.add(n1)
            for key, tris in linjie.items():
                if key in self.sAgentSet:
                    for n1, n2, n3 in tris:
                        if n1 in self.iAgentSet and n2 in self.iAgentSet and random.random() <= beta2:
                            newI.add(key); break
                        if n1 in self.iAgentSet and n3 in self.iAgentSet and random.random() <= beta2:
                            newI.add(key); break
                        if n2 in self.iAgentSet and n3 in self.iAgentSet and random.random() <= beta2:
                            newI.add(key); break
            for n1, n2, n3 in self.triangles_list:
                if n1 in self.iAgentSet and n2 in self.sAgentSet and n3 in self.sAgentSet and random.random() <= 2 * beta2:
                    newR1.add(n1)
                if n2 in self.iAgentSet and n1 in self.sAgentSet and n3 in self.sAgentSet and random.random() <= 2 * beta2:
                    newR1.add(n2)
                if n3 in self.iAgentSet and n1 in self.sAgentSet and n2 in self.sAgentSet and random.random() <= 2 * beta2:
                    newR1.add(n3)
            for n in newI:
                self.infectAgent(n)
            newR = set(newR1)
            if len(self.iAgentSet) < self.N:
                for rec in list(self.iAgentSet):
                    if rec in newI:
                        continue
                    if random.random() <= mu:
                        newR.add(rec)
            for n in newR:
                self.recoverAgent(n)
            self.iList.append(len(self.iAgentSet))
            self.t += 1
        return self.iList

    def get_stationary_rho(self, normed=True, last_k_values=30):
        if not self.iList:
            return 0
        x = np.array(self.iList, dtype=float)
        if normed:
            x /= self.N
        if x[-1] == 1:
            return 1
        if x[-1] == 0:
            return 0
        v = np.nanmean(x[-last_k_values:])
        return float(np.nan_to_num(v))

def build_neighbors_from_ei(ei, triangles_nodes=None):
    if isinstance(ei, (list, tuple)) and len(ei) == 2 and isinstance(ei[0], int):
        n_nodes, edges = ei[0] + 1, ei[1]
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges or [])
    else:
        edges = ei or []
        nodes = set()
        for u, v in edges:
            nodes.add(u); nodes.add(v)
        if triangles_nodes:
            nodes.update(triangles_nodes)
        G = nx.Graph()
        G.add_nodes_from(sorted(nodes))
        G.add_edges_from(edges)
    d = {}
    for n in G.nodes():
        d[n] = set(G.neighbors(n))
    return d

def sample_fraction(nodes, frac, rng, exclude=None):
    base = list(set(nodes) - (exclude or set()))
    k = int(round(frac * len(nodes)))
    k = max(0, min(len(base), k))
    return set(rng.sample(base, k)) if k > 0 else set()

def run_one_simulation(args):
    it_num, beta1s, beta2, t_max, mu, neighbors_dict, triangles_list, linjie, infect_frac, immune_frac, seed = args
    rng = random.Random(None if seed is None else (seed + it_num))
    nodes = list(neighbors_dict.keys())
    immune = sample_fraction(nodes, immune_frac, rng)
    infected_seed = sample_fraction(nodes, infect_frac, rng, exclude=immune)
    rhos = []
    for beta1 in beta1s:
        m = SimplagionModel(neighbors_dict, triangles_list)
        m.initial_setup(infected_seed, immune)
        m.run(t_max, beta1, beta2, mu, linjie)
        rhos.append(m.get_stationary_rho(normed=True, last_k_values=30))
    return rhos

if __name__ == '__main__':
    triangles_list = []
    linjie = {}
    ei = []
    triangles_nodes = set().union(*triangles_list) if triangles_list else set()
    neighbors_dict = build_neighbors_from_ei(ei, triangles_nodes=triangles_nodes)
    mu = 0.5
    lambda1s = np.linspace(0.0, 2.0, 21)
    beta1s = (mu * lambda1s) / 6.0
    t_max = 1000
    n_simulations = 20
    n_processes = 4
    lambda2 = 0.8
    beta2 = (mu * lambda2) / 3.0
    infect_frac = 0.02
    immune_frac = 0.0
    args = []
    for it in range(n_simulations):
        args.append([it, beta1s, beta2, t_max, mu, neighbors_dict, triangles_list, linjie, infect_frac, immune_frac, 1234])
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_one_simulation, args)
    print(results)
