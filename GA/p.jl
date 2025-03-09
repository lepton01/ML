#Angel Ortiz
#28/02/2023
using Statistics, LinearAlgebra, Random, GLMakie
"""
    Test function to optimize.
"""
f(x::Real, y::Real) = (x - 3)^2 + (y + 5)^2
function fitnessF32(P::Array{Float32})
    v = Vector{Float32}(undef, size(P, 1))
    @views for ii in eachindex(v)
        v[ii] = f(P[ii, 1], P[ii, 2])
    end
    return v
end

function crossF32(S1::Vector{Float32}, S2::Vector{Float32}, M::Int32)
    @views for _ in 1:M
        xp = ceil(Int32, 2rand())
        ω = rand(Float32)
        mama = ceil(Int32, 2M * rand())
        papa = ceil(Int32, 2M * rand())
        if xp == 1
            push!(S1, Float32(1 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[papa])
            push!(S1, Float32(1 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[mama])
        elseif xp == 2
            push!(S2, Float32(1 - ω) * S2[papa] + ω * S2[mama])
            push!(S1, S1[papa])
            push!(S1, Float32(1 - ω) * S1[mama] + ω * S1[papa])
            push!(S2, S2[mama])
        end
    end
    return S1, S2
end

function mainF32(pop::Integer, mut::Float32; para::Integer=2, max_i::Int32=10::Int32)
    var_min = -50
    var_max = 50
    P::Array{Float32,2} = (var_max - var_min) * randn(Float32, (pop, para)) .+ var_min
    sel::Float32 = 0.5
    keep = floor(Int32, sel * pop)
    M = round(Int32, (pop - keep) / 2)
    n_mut = ceil(Int32, (pop - 1) * mut * para)
    pob_save = P
    cuenta = 1
    for _ in 1:max_i
        fit = fitnessF32(P)
        S1 = Float32[]
        S2 = Float32[]
        cc::Int32 = 1
        @views while length(S1) < floor(Int32, pop / 2)
            if cc < pop && fit[cc] <= median(fit)
                push!(S1, P[cc, 1])
                push!(S2, P[cc, 2])
            elseif cc == pop
                a = ϵ = ceil(Int32, pop * rand(Float32))
                b = ϵ = ceil(Int32, pop * rand(Float32))
                push!(S1, P[a, 1])
                push!(S2, P[b, 2])
            end
            cc += 1
        end
        v1 = Float32[]
        v2 = Float32[]
        v1, v2 = crossF32(S1, S2, M)
        for _ in 1:n_mut
            ϵ = ceil(Int32, length(S1) * rand())
            ζ = randn(Float32)
            v1[ϵ], v2[ϵ] = Float32(1 - 0.2ζ) * v1[ϵ], Float32(1 - 0.2ζ) * v2[ϵ]
        end
        P = hcat(v1, v2)
        if cuenta != 1
            pob_save = vcat(pob_save, P)
        end
        cuenta += 1
    end
    return P, pob_save
end
mainF32(pop::Integer, mut::Float64; para::Integer=2, max_i::Int=10) = mainF32(pop, mut |> Float32; para=para, max_i=Int32(max_i))

##
@btime P, p = mainF32(10, 0.3; max_i=10)
f1 = Figure()
ax1 = Axis(f1[1, 1], title="Evolution, solution = ($(mean(view(P, :, 1))),$(mean(view(P, :, 2))))", aspect=1)
heatmap!(ax1, -8:0.01:13, -15:0.01:5, log10.(f.(-8:0.01:13, (-15:0.01:5)')))
scatter!(ax1, view(p, :, 1), view(p, :, 2))
f1

## performance
#@code_warntype main(20, 0.3, max_i=1000)
#using ProfileView
#@profview main(20, 0.3, max_i=1000)

##
using BenchmarkTools
a = rand(Float64, (1000, 2))
b = @benchmark fitness($a)
c = @benchmark @views f.($a[:, 1], $a[:, 2])
