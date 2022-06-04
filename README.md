# DataDrivenCSLS.jl

<p align="center">
  <img width="600" alt="Screenshot 2022-06-04 at 14 42 15" src="https://user-images.githubusercontent.com/45042779/171999438-d44997ef-6832-47ad-8f8a-7223aa64de18.png">
</p>

This module implements **data-driven techniques for finding probabilistic stability guarantees for hybrid systems** whose switching rule can be modelled with an automaton [BWJ22a, BWJ22b]. These systems are called constrained switching linear systems (CSLSs).

## Contents

<code>DataDrivenCSLS.jl</code> module is composed of the following files:
* <code>bounds.jl</code>: code that implements techniques described in [BWJ22a] and [BWJ22b] to compute **probabilistic stability guarantees for CSLSs from a finite set of sampled trajectories**
* <code>trajectories.jl</code>: contains tools to generate random trajectories for a given CSLS
* <code>whitebox.jl</code>: contains white-box tools to approximate the joint spectral radius, constrained joint spectral radius, and allows to design a white-box LQR controller for CSLSs

## Example

```julia
include("/src/DataDrivenCSLS.jl")
using Main.DataDrivenCSLS

# Definition of the CSLS S(G, Σ)
Σ = [
  [-0.12 -0.51; -0.51 0.76],
  [-1.00  0.35; -0.01 0.51]
]
G = LightAutomaton(2)
add_transition!(G, 1, 1, 1)
add_transition!(G, 1, 2, 2)
add_transition!(G, 2, 1, 2)

# Simulation of N 1-step samples
l = 1
N = 10000
u, v, x, y = generate_trajectories(Σ, G, N, l)

# In the black-box setting: computation of 95%-sure upper bounds and lower bounds on the CJSR from data
m = size(Σ)[1]
V = nstates(G)
y = reshape(y, l, size(x))
lb_MQLF, ub_MQLF = bounds_MQLF(x, u, y, v, V, .975, .975, m)
```

## Sources

This module was created in the context of my master's thesis at UCLouvain:

**Title** "Learning stability guarantees for black-box hybrid systems"\
**Subtitle** "From arbitrary to constrained switching linear systems: a step towards complexity"\
**Supervisor** Raphaël M. Jungers

[BWJ22a] Adrien Banse, Zheming Wang, and Raphaël M. Jungers. Learning stability guarantees for data-driven constrained switching linear systems. _25th International Symposium on Mathematical Theory of Networks and Systems_ (Bayreuth, Germany). Submitted (pre-print: https://arxiv.org/abs/2205.00696)

[BWJ22b] Adrien Banse, Zheming Wang, and Raphaël M. Jungers. Black-box stability analysis of hybrid systems with sample-based multiple Lyapunov functions. _61st IEEE Conference on Decision and Control_ (Cancùn, Mexico). Submitted (pre-print: https://arxiv.org/abs/2205.00699)
