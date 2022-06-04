include(joinpath(@__DIR__, "../../src/DataDrivenCSLS.jl"))
using Main.DataDrivenCSLS

CJSR = .95
A1 = [-0.12264385232852724 -0.5132986766380205; -0.512239944583497 0.7657920763290312] * CJSR
A2 = [-1.0025775249390196 0.35564172966144936; -0.01094819510863365 0.5106100376225976] * CJSR
Σ = [A1, A2]

G = LightAutomaton(2)
add_transition!(G, 1, 1, 1)
add_transition!(G, 1, 2, 2)
add_transition!(G, 2, 1, 2)
hs = discreteswitchedsystem(Σ, G)
nothing

VERBOSE = false

N = 10000
N_step = 100
N_begin = 100
N_range = N_begin:N_step:N

# Known information
V = nstates(hs.automaton)
m = size(Σ)[1]

# Simulations
u, v, x, y = generate_trajectories(Σ, G, N, 1)
y = reshape(y, size(x))

# Parameters
β = .99
β1 = (β + 1.) / 2.
β2 = β1

f = open("res_MQLF.txt", "w")
display("Simulations start")
for n = N_range
    time = @elapsed lower_bound, upper_bound = bounds_MQLF(x[:, 1:n], u[1:n], y[:, 1:n], v[1:n], V, β1, β2, m)
    println(f, "$lower_bound $upper_bound")
    if VERBOSE display("(n = $n) done (in $time s): [$lower_bound, $upper_bound]") end
end
close(f)


VERBOSE = false
l = 1

p_lmin = 1/4
p_lmax = 3/4
quantity = 1/p_lmin
quantity_max = 1/p_lmax
u, v, x, y = generate_trajectories(Σ, G, N, l)
y = reshape(y, size(x))                         # to change if l ≂̸ 1

f = open("res_CQLF.txt", "w")
display("Simulations start")
for n = N_range
    time = @elapsed upper_bound = upper_bound_CQLF(x[:, 1:n], y[:, 1:n], β, l, quantity, quantity_max = quantity_max)
    println(f, "$upper_bound")
    if VERBOSE display("(n = $n) done (in $time s): [-, $upper_bound]") end
end

close(f)

using DelimitedFiles
data_CQLF = readdlm("res_CQLF.txt")
data_MQLF = readdlm("res_MQLF.txt")

keep = map(x -> x != -1, data_CQLF)
keep = reshape(keep, size(N_range))

using PyPlot
figure()
margins(x=0)
fill_between(N_range, ones(size(N_range)), color="grey", alpha = 0.3, label="Stability zone")
axhline(CJSR, linestyle="--", color="k", linewidth = 0.6, label="\$\\rho(G, \\Sigma) = \\rho(\\Sigma)\$")
plot(N_range, data_MQLF[:, 1], "-", label="MQLF lower bound")
plot(N_range[keep], data_CQLF[keep], "-",  label="CQLF upper bound (β = 99%)")
plot(N_range, data_MQLF[:, 2], "-", label="MQLF upper bound (β = 99%)")
xlabel("Number of observations \$N\$")
xscale("log")
ylim((0, 3))
legend()
savefig("comparaison_first_case.pdf")
