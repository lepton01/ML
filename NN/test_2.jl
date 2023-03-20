#17/03/2023

using LinearAlgebra, Statistics, Random
using Plots

#include("datagen.jl")

#X = Float32.(hcat(real, fake))
#Y = vcat(ones(train_size), zeros(train_size))

test1_data = [1 1 2 2 -1 -2 -1 -2; 1 2 -1 0 2 1 -1 -2]
y_ = [0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1]

"""
    Network(layer_dims::Vector, n_layers::Int)

``layer_dims`` is input as a vector of integers defining number of neurons in each layer. The first and last should match the size of intup and output.
"""
mutable struct Network
    layer_dims::Vector{Int}
end

"""
    Layer(W, b, Z, cache)

`W` is the weights matrix, `b` is the biases vector, and `Z` is the vector of values to enter the activation function,\\
and `cache` is the vector of values to input to the next layer.
"""
mutable struct Layer
    W::Array{Float32}
    b::Vector{Float32}
    Z::Array{Float32}
    cache::Array
end

"""
    sigmoid(x::Float32)

Sigmoid (`σ`) activation function.\\Returns a Float32 number.
"""
sigmoid(x::Float32)::Float32 = 1/(1 + exp(-x))

"""
    sigmoid_back(x::Float32)

'Inverse' sigmoid (`σ`) activation function.\\
Returns a `Float32` number.
"""
function sigmoid_back(x::Float32)::Float32
    s = sigmoid(x)
    s*(1 - s)
end

"""
    ReLU(x::Float32)

Rectified linear unit activation function. max(0, x).\\
Returns a `Float32` number.
"""
ReLU(x::Float32)::Float32 = x > 0 ? x : 0

"""
    ReLU_back(x::Float32)

'Inverse' rectified linear unit activation function.\\
Returns a `Float32` number.
"""
ReLU_back(x::Float32)::Float32 = x > 0 ? 1 : 0

"""
    init_para(net)

Creates a vector storing type ``Layer`` with the parameters (weights, biases, and values) for all neurons in every layer.
"""
function init_para(net::Network)
    para = Layer[]
    for i ∈ 1:length(net.layer_dims)
        i == 1 ? push!(para, Layer(zeros(Float32, (1, 1)), zeros(net.layer_dims[i]), zeros(net.layer_dims[i]), zeros(net.layer_dims[i]))) : push!(para, Layer(rand(Float32, (net.layer_dims[i], net.layer_dims[i - 1])), zeros(net.layer_dims[i]), zeros(net.layer_dims[i]), zeros(net.layer_dims[i])))
    end

    para
end

"""
    fwd_prop(input::Array, para) -> 
``input`` is an array  ``para`` is a ``Vector{Layer}`` containing all the parameters
"""
function fwd_prop!(input::Array, para)
    n = length(para)
    para[1].cache = input
    for i ∈ 2:n
        para[i].Z = muladd(para[i].W, para[i - 1].cache, para[i].b)
        i < n ? para[i].cache = ReLU.(para[i].Z) : para[i].cache = sigmoid.(para[i].Z)
    end

    para
end

function cost(T::Vector, Y::Vector)::Float32
    @assert length(Y) == length(T) "Not the same size."
    l = length(Y)

    cost = -sum(Y.*log.(T) .+ (1 .- Y).*log.(1 .- T))/l
end

function back_prop!(T, Y, para)
    n = length(para)
    @assert length(Y) == length(T) "Not the same size"
    error = Y - T
    para[end].W = para[end].W + error*para[end - 1].cache'
    para[end].b = para[end].b + error

    para
end

function training(X, Y, dims::Vector, n_it::Int = 100)
    Net = Network(dims)
    parameters = init_para(Net)
    if size(X, 1) != 2
        reshape!(X, 2, :)
    end
    if size(Y, 1) != 2
        reshape!(Y, 2, :)
    end
    for i ∈ 1:n_it
        for j ∈ 1:length(X[1, :])
            parameters = fwd_prop!(X[:, j], parameters)
            parameters = back_prop!(parameters[end].cache, Y[:, j], parameters)
        end
        i != n_it ? nothing : show(parameters[end].cache)
    end

    parameters
end

function testing()
    nothing
end

paras  = training(test1_data, y_, [2, 2, 2])