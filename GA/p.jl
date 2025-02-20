#Angel Ortiz
#28/02/2023
using Statistics, LinearAlgebra, Random, GLMakie

#f(x::Real, y::Real) = x*sin(4x) + 1.1y*sin(2y)
#f(x::Real, y::Real) = sqrt(x*y)*sin(x)*sin(y)
#[0, 10]

f(x, y) = (x - 3)^2 + (y + 15)^2
#f(x::Real, y::Real) = x^2 + 2y^2 - 0.3cos(3π * x) - 0.4cos(4π * y) + 0.7
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

#max_i::Int = 10
#max_cost = 999_999_999
"""
    Test function to optimize.
"""
#f(x, y) = x * sin(4x) + 1.1y * sin(2y)

function fitness(P::AbstractArray)
    v = Vector{Float64}(undef, size(P, 1))
    for ii in eachindex(v)
        v[ii] = f(P[ii, 1], P[ii, 2])
    end
    return v
end

function cross(S1::Vector, S2::Vector, M::Int)
    for _ in 1:M
        xp = ceil(Int, 2rand())
        ω = rand()
        mama = ceil(Int, 2M * rand())
        papa = ceil(Int, 2M * rand())
        if xp == 1
            push!(S1, (1.0 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[papa])
            push!(S1, (1.0 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[mama])
        elseif xp == 2
            push!(S2, (1.0 - ω) * S2[papa] + ω * S2[mama])
            push!(S1, S1[papa])
            push!(S1, (1.0 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[mama])
        end
    end
    return S1, S2
end

function main(pop::Int, mut::Float64; para::Int=2, max_i::Int=10)
    var_min = -50
    var_max = 50
    P = (var_max - var_min) * randn(Float64, (pop, para)) .+ var_min
    sel = 0.5
    keep = floor(Int, sel * pop)
    M = round(Int, (pop - keep) / 2)
    n_mut = ceil(Int, (pop - 1) * mut * para)
    pob_save = Float64[]
    pob_save = P
    for ii in 1:max_i
        fit = P |> fitness
        S1 = Float64[]
        S2 = Float64[]
        cc = 1
        while length(S1) < floor(Int, pop / 2)
            if cc < pop && fit[cc] <= median(fit)
                push!(S1, P[cc, 1])
                push!(S2, P[cc, 2])
            elseif cc == pop
                a = ϵ = ceil(Int, pop * rand())
                b = ϵ = ceil(Int, pop * rand())
                push!(S1, P[a, 1])
                push!(S2, P[b, 2])
            end
            cc += 1
        end
        v1 = Float64[]
        v2 = Float64[]
        v1, v2 = cross(S1, S2, M)
        for _ in 1:n_mut
            ϵ = ceil(Int, length(S1) * rand())
            ζ = randn()
            v1[ϵ], v2[ϵ] = (1 - 0.3ζ) * v1[ϵ], (1 - 0.3ζ) * v2[ϵ]
        end
        P = hcat(v1, v2)
        if ii != 1
            pob_save = vcat(pob_save, P)
        end
    end
    return P, pob_save
end

##
@time P, p = main(20, 0.3, max_i=1000)
f1 = Figure()
ax1 = Axis(f1[1, 1], title="Evolution")
heatmap!(ax1, -13:0.01:18, -35:0.01:5, log10.(f.(-13:0.01:18, (-35:0.01:5)')))
scatter!(ax1, p[:, 1], p[:, 2])
f1
