#06/03/2023

using Statistics, Random, LinearAlgebra
Random.seed!(1)

f(x, y) = x*sin(4x) + 1.1y*sin(2y)

function fitness(P::AbstractArray)
    F = Vector{Float64}(undef, length(P[:, 1]))
    F[1] = f(P[1, 1], P[1, 2])
    p_b = 1

    for i in 2:length(P[:, 1])
        F[i] = f(P[i, 1], P[i, 2])
        if i >= 2 && F[i] < F[i - 1]
            p_b = i
        end
    end
    F, p_b
end

function mov(P::AbstractArray, fit::Vector, best::Int)
    V = Array{Float64}(undef, size(P))

end

function PSOmin(pop::Int, para::Int = 2)
    c1::Float64 = 0.49445
    c2::Float64 = 0.49445

    max_i::Int = 100

    vmax = 0.5
    vmin = -0.5

    popmax::Int = 2
    popmin::Int = -2

    P = randn(Float64, (pop, para))
    V = zeros(Float64, (pop, para))

    
end

function PSOmax(pop::Int, para::Int = 2)
    c1::Float64 = 0.49445
    c2::Float64 = 0.49445

    max_i::Int = 100

    vmax = 0.5
    vmin = -0.5

    popmax::Int = 2
    popmin::Int = -2

    P = randn(Float64, (pop, para))
    V = randn(Float64, (pop, para))

    for i in 1:max_i
        fit, p_best = fitness(P)
        S1 = Float64[]
        S2 = Float64[]
        
    end
    P
end