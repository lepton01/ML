#09/04/2023
using LinearAlgebra, Statistics, Random
using Plots
using Flux, SpecialFunctions, BSON
using Flux: mae
include("bessel_train.jl")
include("model_creation.jl")

s::String = "model_1"
#bessel_model_creation(s)
#@time bessel_train!(collect(Float32, LinRange(0.01, 50, 1000)), 50*rand32(1000), s)

x_test::Float32 = 1.
a_test::Float32 = 0.

"""
    bessel_approx(x, a, s)

Approximates the first kind Bessel function centered at ``a`` given ``x``. `s` determines the model to use.
"""
function bessel_approx(x::AbstractFloat, a::AbstractFloat, model_name::String)
    BSON.@load model_name*".bson" model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X)[end]
    return v, v - besselj(a, x)
end
#@time appx11 = bessel_approx(x_test, a_test, s)

"""
    bessel_approx_gpu(x, a, model_name)

Approximate the first kind Bessel function centered at `a` given `x`. `model_name` determines the model to use.\\
Uses CUDA to compute on the GPU.
"""
function bessel_approx_gpu(x::AbstractFloat, a::AbstractFloat, model_name::String)
    BSON.@load model_name*".bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X |> gpu) |> cpu
    return v[end], v[end] - besselj(a, x)
end
#@time appx12 = bessel_approx_gpu(x_test, a_test, s)
