#09/04/2023
include("bessel_1.jl")

xg::Float32 = rand32()
ag::Float32 = rand32()

# Compute the real value for comparison.
besselReal = besselj(ag, xg)

"""
    bessel_model1(x, a)

Approximates the first kind Bessel function centered at ``a`` given ``x``.
"""
function bessel_model1(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j.bson" model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    return model(X)[end], besselj(x, a)
end
#@time appx11 = bessel_model1(xg, ag)
#appx11 - besselReal

"""
    bessel_model1_gpu(x, a)

Approximates the first kind Bessel function centered at ``a`` given ``x``.\\
Uses CUDA to compute on the GPU.
"""
function bessel_model1_gpu(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j.bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X |> gpu) |> cpu
    return v[end], besselj(x, a)
end
#@time appx12 = bessel_model1_gpu(xg, ag)
#appx12 - besselReal

"""
    bessel_model2(x, a)

Approximates the first kind Bessel function centered at ``a`` given ``x``.
"""
function bessel_model2(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j_2.bson" model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    return model(X)[end], besselj(x, a)
end
#@time appx21 = bessel_model1(xg, ag)
#appx21 - besselReal

"""
    bessel_model2_gpu(x, a)

Approximates the first kind Bessel function centered at ``a`` given ``x``.\\
Uses CUDA to compute on the GPU.
"""
function bessel_model2_gpu(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j_2.bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X |> gpu) |> cpu
    return v[end], besselj(x, a)
end
#@time appx22 = bessel_model3(xg, ag)
#appx22 - besselReal

"""
    bessel_model1(x, a)

Approximates the first kind Bessel function centered at ``a`` given ``x``.
"""
function bessel_model3(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j_3.bson" model
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    return model(X)[end], besselj(x, a)
end
#@time appx31 = bessel_model3(xg, ag)
#appx31 - besselReal

"""
    bessel_approx3_gpu(x, a)

Approximates the first kind Bessel function centered at ``a`` given ``x``.\\
Uses CUDA to compute on the GPU.
"""
function bessel_model3_gpu(x::AbstractFloat, a::AbstractFloat)
    BSON.@load "bessel_j_3.bson" model
    model = model |> gpu
    X = Array{Float32}(undef, (2, 1))
    X[:, 1] = [x, a]
    v = model(X |> gpu) |> cpu
    return v[end], besselj(x, a)
end
#@time appx32 = bessel_model3_gpu(xg, ag)
#appx32 - besselReal
