#Angel Ortiz
#06/03/2023
using Statistics, Random, LinearAlgebra

f(x::Real, y::Real) = x * sin(4x) + 1.1y * sin(2y)
#=
function fitness(P::AbstractArray, L::AbstractArray, g_best, g_best_fit)
    F = Vector{Float64}(undef, length(P[:, 1]))
    F[1] = f(P[1, 1], P[1, 2])

    for i in 2:length(P[:, 1])
        F[i] = f(P[i, 1], P[i, 2])
        if F[i] < F[pop_best]
            pop_best = i
        end
    end
    F, pop_best
end

function mov(P::AbstractArray, V::AbstractArray, fit::Vector, p_best::Int)
    nothing
end
=#

function PSOmax(pop_size::Int, para::Int=2)
    nothing
end

function PSOmin(pop_size::Int, para::Int=2)
    c1::Float64 = 0.49445
    c2::Float64 = 0.49445

    max_i::Int = 100

    v_max = 0.5
    v_min = -0.5

    pop_max::Int = 20
    pop_min::Int = -20

    P = (pop_max - pop_min) * rand(Float64, (pop_size, para)) .+ pop_min
    V = (v_max - v_min) * rand(Float64, (pop_size, para)) .+ v_min
    L = P
    g_best = (pop_max - pop_min) * rand(Float64, para) .+ pop_min
    for i in 1:pop_size
        if f(P[i, 1], P[i, 2]) < f(g_best[1], g_best[2])
            g_best = P[i, :]
        end
    end

    for _ in 1:max_i
        #fit, g_best = fitness(P, L, g_best, g_best_fit)
        for jj in 1:pop_size
            V[jj, 1] = V[jj, 1] + c1 * rand() * (L[jj, 1] - P[jj, 1]) + c2 * rand() * (g_best[1] - P[jj, 1])
            V[jj, 2] = V[jj, 2] + c1 * rand() * (L[jj, 2] - P[jj, 2]) + c2 * rand() * (g_best[2] - P[jj, 2])
            P[jj, 1] = P[jj, 1] + V[jj, 1]
            P[jj, 2] = P[jj, 2] + V[jj, 2]
            if f(P[jj, 1], P[jj, 2]) < f(L[jj, 1], L[jj, 2])
                L[jj, 1], L[jj, 2] = P[jj, 1], P[jj, 2]
                if f(P[jj, 1], P[jj, 2]) < f(g_best[1], g_best[2])
                    g_best = P[jj, :]
                end
            end
        end
    end
    return P
end
