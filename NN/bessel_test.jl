#09/04/2023
include("bessel_train.jl")
include("model_creation.jl")

s::String = "bessel_j_7.bson"
#@time bessel_model_creation(collect(Float32, LinRange(0.01, 50, 1000)), 50*rand32(1000), s)
#@time bessel_train(collect(Float32, LinRange(0.01, 50, 1000)), 50*rand32(1000), s)

xg::Float32 = 2.
ag::Float32 = 1.

"""
    bessel_model(x, a, s)

Approximates the first kind Bessel function centered at ``a`` given ``x``. `s` determines the model to use.
"""
function bessel_model(x::AbstractFloat, a::AbstractFloat, model_name::String)
    BSON.@load model_name model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X)[end]
    return v, v - besselj(a, x)
end
#@time appx11 = bessel_model(xg, ag, s)

"""
    bessel_model_gpu(x, a)

Approximates the first kind Bessel function centered at ``a`` given ``x``. `s` determines the model to use.\\
Uses CUDA to compute on the GPU.
"""
function bessel_model_gpu(x::AbstractFloat, a::AbstractFloat, model_name::String)
    BSON.@load model_name model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X |> gpu) |> cpu
    return v[end], v[end] - besselj(a, x)
end
#@time appx12 = bessel_model_gpu(xg, ag, s)
