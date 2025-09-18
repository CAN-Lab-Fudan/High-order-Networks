# Higher-order-Networks

## 1. Higher-order Epidemic Spreading in Simplicial Networks
### 1.1 Background Information
Epidemics do not only spread through simple pairwise contacts (like families or colleagues), but also through group interactions such as classrooms, workplaces, or communities. These higher-order structures can be modeled using simplicial complexes, which capture multi-node interactions beyond traditional network edges.

In this work, we propose a **higher-order** epidemic model that:
* Ensures node state consistency within simplices (groups).
* Captures the coexistence of interacting groups and their dynamic effects.
* Provides a solvable network model capable of showing bistable regions and epidemic thresholds.

Our paper is available at this link [this link](https://ieeexplore.ieee.org/document/11145328). Citations are welcome.

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
   * **Intput**: list/iterable of undirected edges (u, v) (int), u != v.
      
   * **Output**: Cliques_by_k: dict mapping k (nodes) → list of k-node cliques (sorted tuples).
  
