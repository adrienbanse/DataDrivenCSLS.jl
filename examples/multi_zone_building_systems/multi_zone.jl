using LinearAlgebra
using SwitchOnSafety
using ControlSystems
using StaticArrays
using HybridSystems
using JuMP
using MosekTools
using Random
using IncGammaBeta

dim = 3; dim_in = 3
c = 1375
cp = 1.012
R12 = 1.5
Ro12 = 3
Ro3 = 2.7
tau = 3 * 60
m = 0.14
T_target = 24
Ts = 16
To = 32
q = [.1, .1, .12]
T_0 = [38, 34, 32]

Σ_basis = []
for R13 in [1.2 0.8]
    for R23 in [1.2 0.8]
        A = [
            1-tau/c*(1/R12+1/R13+1/Ro12)    tau/c/R12                       tau/c/R13;
            tau/c/R12                       1-tau/c*(1/R12+1/R23+1/Ro12)    tau/c/R23;
            tau/c/R13                       tau/c/R23                       1-tau/c*(1/R13+1/R23+1/Ro3)
        ]
        push!(Σ_basis, A)
    end
end

B = tau / c * Matrix(I, dim_in, dim_in)

include("../../src/WhiteBox.jl")
Q = Matrix(I, dim, dim)
R = 0.02 * Matrix(I, dim_in, dim_in)
K, P = white_box_LQR(Σ_basis, B, Q, R)

# 1 and 2 don't work
K0 = copy(K)
K0[:, 1] .= 0
K0[:, 2] .= 0
# 1 works, 2 doesn't work
K1 = copy(K)
K1[:, 2] .= 0
# 1 doesn't work, 2 works
K2 = copy(K)
K2[:, 1] .= 0

function charac_to_idx(charac)
    if charac[1] && charac[2]
        return 1
    elseif charac[2]
        return 2
    elseif charac[1]
        return 3
    end
    return 4
end

function maps_to_σ(failed, closed)
    row_idx = charac_to_idx(failed)
    col_idx = charac_to_idx(closed)
    4 * (row_idx - 1) + col_idx 
end

Σ_non_fail = []
for i = 1:size(Σ_basis)[1]
    push!(Σ_non_fail, copy(Σ_basis[i]) + B * K)
end

Σ_fail = []
for Kσ in [K0, K1, K2, K]
    for i = 1:size(Σ_basis)[1]
        push!(Σ_fail, copy(Σ_basis[i]) + B * Kσ)
    end
end

G = LightAutomaton(4)

add_transition!(G, 1, 1, maps_to_σ([false, false], [true, true]))
add_transition!(G, 1, 2, maps_to_σ([false, false], [false, true]))
add_transition!(G, 1, 3, maps_to_σ([false, false], [true, false]))
add_transition!(G, 1, 4, maps_to_σ([false, false], [false, false]))

add_transition!(G, 2, 1, maps_to_σ([true, false], [true, true]))
add_transition!(G, 2, 2, maps_to_σ([true, false], [false, true]))
add_transition!(G, 2, 3, maps_to_σ([true, false], [true, false]))
add_transition!(G, 2, 4, maps_to_σ([true, false], [false, false]))

add_transition!(G, 3, 1, maps_to_σ([true, true], [true, true]))
add_transition!(G, 3, 2, maps_to_σ([true, true], [false, true]))
add_transition!(G, 3, 3, maps_to_σ([true, true], [true, false]))
add_transition!(G, 3, 4, maps_to_σ([true, true], [false, false]))

add_transition!(G, 4, 1, maps_to_σ([false, true], [true, true]))
add_transition!(G, 4, 2, maps_to_σ([false, true], [false, true]))
add_transition!(G, 4, 3, maps_to_σ([false, true], [true, false]))
add_transition!(G, 4, 4, maps_to_σ([false, true], [false, false]))

include("../../src/Trajectories.jl")
include("../../src/Bounds.jl")

# Known information
V = nstates(G)
m = size(Σ_fail)[1]

# Simulations
N = 50000
u, v, x, y = generate_trajectories(Σ_fail, G, N, 1)
y = reshape(y, size(x))

# Parameters
β = .95
β1 = 1. - 2. * (1. - β)
β2 = β1

f = open("bounds.txt", "w")

println("Simulations start")
for n = 1000:5000:N
    time = @elapsed lower_bound, upper_bound = bounds_MQLF(x[:, 1:n], u[1:n], y[:, 1:n], v[1:n], V, β1, β2, m)
    write(f, "$lower_bound $upper_bound\n")
    println("(n = $n) done (in $time s)")
end
close(f)

