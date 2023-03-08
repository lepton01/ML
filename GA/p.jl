#28/02/2023

using Plots
using Statistics, LinearAlgebra, Random
Random.seed!(1)



f(x, y) = x*sin(4x) + 1.1y*sin(2y)

function fitness(P::AbstractArray)
    v = Vector{Float64}(undef, length(P[:, 1]))

    for i in eachindex(v)
        v[i] = f(P[i, 1], P[i, 2])
    end
    v
end

function cross(S1::Vector, S2::Vector, M::Int)
    for i in 1:M
        xp = Int(ceil(2rand()))
        α = rand()
        ma = Int(ceil(2M*rand()))
        pa = Int(ceil(2M*rand()))

        if xp == 1
            push!(S1, (1.0 - α)*S1[ma] + α*S1[pa])

            push!(S2, S2[pa])

            push!(S1, (1.0 - α)*S1[ma] + α*S1[pa])

            push!(S2, S2[ma])
        elseif xp == 2
            push!(S2, (1.0 - α)*S2[pa] + α*S2[ma])

            push!(S1, S1[pa])

            push!(S1, (1.0 - α)*S1[ma] + α*S1[pa])

            push!(S2, S2[ma])
        end
    end
    S1, S2
end

function main(pop::Int, mut::Float64, para::Int = 2)
    max_i::Int = 1000
    max_cost = 999_999_999
    var_min = -10
    var_max = 20
    P = (var_max - var_min)*randn(Float64, (pop, para)) .+ var_min
    sel = 0.5
    keep = Int(floor(sel*pop))
    M = Int(round((pop - keep)/2))
    n_mut = Int(ceil((pop - 1)*mut*para))

    for i in 1:max_i
        fit = fitness(P)
        S1 = Float64[]
        S2 = Float64[]
        c = 1
        while length(S1) < Int(floor(pop/2))
            if c < pop && fit[c] <= median(c)
                push!(S1, P[c, 1])
                push!(S2, P[c, 2])
            elseif c == pop
                a = ϵ = Int(ceil(pop*rand()))
                b = ϵ = Int(ceil(pop*rand()))
                push!(S1, P[a, 1])
                push!(S2, P[b, 2])
            end
            c += 1
        end
        v1 = Float64[]
        v2 = Float64[]
        v1, v2 = cross(S1, S2, M)
        for q in 1:n_mut
            ϵ = Int(ceil(length(S1)*rand()))
            ζ = randn()
            v1[ϵ], v2[ϵ] = (1 - 0.1ζ)*v1[ϵ], (1 - 0.1ζ)*v2[ϵ]
        end
        P = hcat(v1, v2)
    end
    P
end
