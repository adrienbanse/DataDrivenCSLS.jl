using LinearAlgebra

include("../../src/Trajectories.jl")
m = 2
dim_max = 8
f = open("matrices.txt", "a")
for dim = 5:5
    for _ in 1:m
        println(f, generate_random_matrix(dim))
        println(f, "")
    end
end
close(f)
