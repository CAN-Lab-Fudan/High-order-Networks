# Higher-order-Networks

![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg) ![Python Version](https://img.shields.io/badge/python-%3E%3D%203.7-blue.svg) 
 ![License](https://img.shields.io/badge/license-Apache%202.0-green.svg) ![Platform](https://img.shields.io/badge/platform-win%20%7C%20macos%20%7C%20linux-yellow.svg)

## 1. Higher-order Epidemic Spreading in Simplicial Networks
### 1.1 Background Information
Epidemics do not only spread through simple pairwise contacts (like families or colleagues), but also through group interactions such as classrooms, workplaces, or communities. These higher-order structures can be modeled using simplicial complexes, which capture multi-node interactions beyond traditional network edges.

In this work, we propose a **higher-order** epidemic model that:
* Ensures node state consistency within simplices (groups).
* Captures the coexistence of interacting groups and their dynamic effects.
* Provides a solvable network model capable of showing bistable regions and epidemic thresholds.

Our paper is available at [this link](https://ieeexplore.ieee.org/document/11145328). Citations are welcome.

```bibtex
@ARTICLE{11145328,
  author={Zhao, Yang and Li, Cong and Shi, Dinghua and Chen, Guanrong and Li, Xiang},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems}, 
  title={Higher Order Epidemic Spreading in Simplicial Networks}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Epidemics;Mathematical models;Analytical models;Aerodynamics;Numerical models;Diseases;Couplings;Tensors;Silicon;Nonlinear dynamical systems;Epidemic spreading;higher order structure;simplicial network;state consistency},
  doi={10.1109/TSMC.2025.3601832}}
```
### 1.2 Key Features
<img src="https://github.com/CAN-Lab-Fudan/Higher-order-Networks/blob/master/Higher-order%20Epidemic%20Spreading%20in%20Simplicial%20Networks/Key%20Features.jpg" width="950px">

Our tool includes the following core functionalities:
* **Theoretical Analysis**: Derived epidemic thresholds and bistable regions in various simplicial networks (random regular, simplex-lattice, star, and heterogeneous).
* **Simulations**: Monte Carlo experiments confirm phase transitions and bistability due to higher-order interactions.
* **Inter-simplex Interactions**: Show that allowing interactions between different groups lowers the epidemic threshold and enlarges the bistable region.
* **Practical Insight**: Epidemic dynamics are highly sensitive to the density and location of initially infected nodes.

### 1.3 How to Run
#### 1) Prerequisites:
* Python 3.8 or later.
* Required libraries: ``networkx``, ``numpy``, ``matplotlib``, ``math``, ``random``  (``install with pip install networkx, numpy, matplotlib, math, random``).

#### 2) Setup
Clone or download this repository:
```bash
git clone https://github.com/CAN-Lab-Fudan/Higher-order-Networks/tree/master/Higher-order%20Epidemic%20Spreading%20in%20Simplicial%20Networks.git
```

#### 3) Running the Scripts:

Follow these steps to run the tool:

1. **Identify higher-order structures in the network**:

   Enumerate all cliques (simplices) in an undirected graph, from edges (2-cliques) to higher-order (triangles, tetrahedra, …).1.2. Identify all inter-group interactions among higher-order structures

   Run the following command:

   ```Python
   python enumerate_simplices.py
   ```
   
   * **Intput**:
     * Edges: list/iterable of undirected edges (u, v) (int), u != v.
      
   * **Output**:
     * Cliques_by_k: dict mapping k (nodes) → list of k-node cliques (sorted tuples).

2. **Identify inter-group interactions among higher-order structures**:

   For each node, list the higher-order neighbor triangles reachable via its incident edges, under intersection-aware semantics.

   Run the following command:

   ```Python
   python hone.py
   ```
   
   * **Intput**:
     * Edges: list/iterable of undirected edges (u, v).
     * Triangles: list of base triangles (a, b, c) already found (sorted or not).

   * **Output**:
     * Dict node -> List[triangle]. Each triangle appears once per node by default

3. **Construction of Simplicial Networks with Different Topologies**:

   Generate edges only for average number of linked triangles of a node contained in a simplex ⟨k<sub>∇</sub>⟩ and average degree ⟨k⟩.

   Run the following command:

   ```Python
   python simplicial_edge_generators.py
   ```

   * **Intput**:
     * Number of cliques: N_2.
     * Average degree of cliques: ⟨k<sub>∇</sub>⟩.
     * Average degree: ⟨k⟩.

   * **Output**:
     * Link probability: p_1.
     * Link probability of two simple: <sub>∇</sub>.
     * Edges: a set of undirected pairs (u, v) with u < v.

4. **Analysis of Epidemic Dynamics on Simplicial Networks**：

   A compact simulator for a simplicial SIS process with: pairwise SIS on an input edge set (``ei``), intra-group promotion/inhibition on ``triangles_list``, and inter-group promotion via ``linjie``.

   Run the following command:

   ```Python
   python pb_sis2_mft.py，Simplicial SIS model.py
   ```

   * **Intput**:
     * Different network topologies and dynamical parameters.

   * **Output**:
     * Number of infected nodes over time for each run and estimated from the tail average of I(t)/N.

<img src="https://github.com/CAN-Lab-Fudan/Higher-order-Networks/blob/master/Higher-order%20Epidemic%20Spreading%20in%20Simplicial%20Networks/Epidemic%20outcomes.jpg" width="750px">
   
   
