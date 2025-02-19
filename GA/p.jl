#Angel Ortiz
#28/02/2023
using Statistics, LinearAlgebra, Random, GLMakie

#f(x::Real, y::Real) = x*sin(4x) + 1.1y*sin(2y)
#f(x::Real, y::Real) = sqrt(x*y)*sin(x)*sin(y)
#[0, 10]

f(x::Real, y::Real) = x^2 + 2y^2 - 0.3cos(3π * x) - 0.4cos(4π * y) + 0.7
#f(x::Real, y::Real) = x^2 + 2y^2 - 0.3cos(3π*x)*0.4cos(4π*y) + 0.3
#f(x::Real, y::Real) = x^2 + 2y^2 - 0.3cos(3π*x + 4π*y) + 0.3
#[-100, 100]

#f(x::Real, y::Real) = (x + 2y -7)^2 + (2x + y -5)^2
#[-10, 10]

#=
mutable struct crom
    g::AbstractVector
    fs::Float64
end
cr = crom([], 0)
=#

max_i::Int = 1000
#max_cost = 999_999_999
"""
    Test function to optimize.
"""
f(x, y) = x * sin(4x) + 1.1y * sin(2y)

function fitness(P::AbstractArray)
    v = Vector{Float64}(undef, length(P[:, 1]))
    for ii in eachindex(v)
        v[ii] = f(P[ii, 1], P[ii, 2])
    end
    return v
end

function cross(S1::Vector, S2::Vector, M::Int)
    for _ in 1:M
        xp = Int(ceil(2rand()))
        α = rand()
        ma = Int(ceil(2M * rand()))
        pa = Int(ceil(2M * rand()))

        if xp == 1
            push!(S1, (1.0 - α) * S1[ma] + α * S1[pa])

            push!(S2, S2[pa])

            push!(S1, (1.0 - α) * S1[ma] + α * S1[pa])

            push!(S2, S2[ma])
        elseif xp == 2
            push!(S2, (1.0 - α) * S2[pa] + α * S2[ma])

            push!(S1, S1[pa])

            push!(S1, (1.0 - α) * S1[ma] + α * S1[pa])

            push!(S2, S2[ma])
        end
    end
    return S1, S2
end

function main(pop::Int, mut::Float64, para::Int=2)
    var_min = -10
    var_max = 20
    P = (var_max - var_min) * randn(Float64, (pop, para)) .+ var_min
    sel = 0.5
    keep = (sel * pop) |> floor |> Int
    M = (round((pop - keep) / 2)) |> Int
    n_mut = (ceil((pop - 1) * mut * para)) |> Int
    for _ in 1:max_i
        fit = P |> fitness
        S1 = Float64[]
        S2 = Float64[]
        c = 1
        while length(S1) < Int(floor(pop / 2))
            if c < pop && fit[c] <= median(c)
                push!(S1, P[c, 1])
                push!(S2, P[c, 2])
            elseif c == pop
                a = ϵ = Int(ceil(pop * rand()))
                b = ϵ = Int(ceil(pop * rand()))
                push!(S1, P[a, 1])
                push!(S2, P[b, 2])
            end
            c += 1
        end
        v1 = Float64[]
        v2 = Float64[]
        v1, v2 = cross(S1, S2, M)
        for _ in 1:n_mut
            ϵ = Int(ceil(length(S1) * rand()))
            ζ = randn()
            v1[ϵ], v2[ϵ] = (1 - 0.1ζ) * v1[ϵ], (1 - 0.1ζ) * v2[ϵ]
        end
        P = hcat(v1, v2)
    end
    return P
end

##
main(10,0.2)
