#09/04/2023
include("bessel_1.jl")

xg::Float32 = 3.0
ag::Float32 = 1.0

# Compute the real value for comparison.
besselReal = besselj(ag, xg)

"""
    bessel_model1(x, a)

Approximates the first kind Bessel function centered at a on a given x.
"""
function bessel_model1(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j.bson" model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    return model(X)[end]
end
#@time appx11 = bessel_model1(1., 2.)
#appx11 - besselReal

"""
    bessel_model1_gpu(x, a)

Approximates the first kind Bessel function centered at a on a given x.\
Uses CUDA to compute on the GPU.
"""
function bessel_model1_gpu(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j.bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X |> gpu) |> cpu
    return v[end]
end
#@time appx12 = bessel_model1_gpu(1., 2.)
#appx12 - besselReal

"""
    bessel_model2(x, a)

Approximates the first kind Bessel function centered at a on a given x.
"""
function bessel_model2(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j_2.bson" model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    return model(X)[end]
end
#@time appx21 = bessel_model1(1., 2.)
#appx21 - besselReal

"""
    bessel_model2_gpu(x, a)

Approximates the first kind Bessel function centered at a on a given x.\
Uses CUDA to compute on the GPU.
"""
function bessel_model2_gpu(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j_2.bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X |> gpu) |> cpu
    return v[end]
end
#@time appx22 = bessel_model3(1., 2.)
#appx22 - besselReal

"""
    bessel_model1(x, a)

Approximates the first kind Bessel function centered at a on a given x.
"""
function bessel_model3(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j_3.bson" model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    return model(X)[end]
end
#@time appx31 = bessel_model3(1., 2.)
#appx31 - besselReal

"""
    bessel_approx3_gpu(x, a)

Approximates the first kind Bessel function centered at a on a given x.\
Uses CUDA to compute on the GPU.
"""
function bessel_model3_gpu(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j_3.bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X |> gpu) |> cpu
    return v[end]
end
#@time appx32 = bessel_model3_gpu(1., 2.)
#appx32 - besselReal
