#Angel Ortiz
#28/02/2023
using Statistics, LinearAlgebra, Random, GLMakie
"""
    Test function to optimize.
"""
f(x::Real, y::Real) = (x - 3)^2 + (y + 5)^2
function fitness(P::Array{<:Real})
    v = Vector{Float64}(undef, size(P, 1))
    @inbounds @views for ii ∈ eachindex(v)
        v[ii] = f(P[ii, 1], P[ii, 2])
    end
    return v
end

function cross(S1::Vector{<:Real}, S2::Vector{<:Real}, M::Integer)
    @inbounds @views for _ ∈ 1:M
        xp = ceil(2 * rand())
        ω = rand()
        mama = ceil(Int, 2 * M * rand())
        papa = ceil(Int, 2 * M * rand())
        if xp == 1
            push!(S1, (1 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[papa])
            push!(S1, (1 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[mama])
        elseif xp == 2
            push!(S2, (1 - ω) * S2[papa] + ω * S2[mama])
            push!(S1, S1[papa])
            push!(S1, (1 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[mama])
        end
    end
    return S1, S2
end

function main(pop::Integer, mut; para::Integer=2, max_i::Integer=10)
    var_min = -50
    var_max = 50
    P = (var_max - var_min) * randn((pop, para)) .+ var_min
    sel = 1 // 2
    keep = floor(sel * pop)
    M = round(Int, (pop - keep) // 2)
    n_mut = ceil(Int, (pop - 1) * mut * para)
    pob_save = P
    @inbounds for ii ∈ 1:max_i
        fit = fitness(P)
        S1 = Float64[]
        S2 = Float64[]
        cc = 1
        @views while length(S1) < floor(Int, pop // 2)
            if cc < pop && fit[cc] <= median(fit)
                push!(S1, P[cc, 1])
                push!(S2, P[cc, 2])
            elseif cc == pop
                a = ceil(Int, pop * rand())
                b = ceil(Int, pop * rand())
                push!(S1, P[a, 1])
                push!(S2, P[b, 2])
            end
            cc += 1
        end
        S1, S2 = cross(S1, S2, M)
        for _ ∈ 1:n_mut
            ϵ = ceil(Int, length(S1) * rand())
            ζ = randn()
            S1[ϵ] = (1 - 1 // 4 * ζ) * S1[ϵ]
            S2[ϵ] = (1 - 1 // 4 * ζ) * S2[ϵ]
        end
        P = hcat(S1, S2)
        if ii != 1
            pob_save = vcat(pob_save, P)
        end
    end
    return P, pob_save
end
main(2, 0; max_i=1);

##
@time P, p = main(40, 0.3; max_i=100);
f1 = Figure()
ax1 = Axis(f1[1, 1], title="Evolution, solution = ($(@views mean(P[:, 1])),$(@views mean(P[:, 2])))", aspect=1)
heatmap!(ax1, -13:0.01:18, -15:0.01:5, log.(f.(-13:0.01:18, (-15:0.01:5)')))
@views scatter!(ax1, P[:, 1], P[:, 2])
Colorbar(f1[1, 2])
f1
