#17/03/2023

using LinearAlgebra, Statistics, Random
using Plots

#include("datagen.jl")

#X = Float32.(hcat(real, fake))
#Y = vcat(ones(train_size), zeros(train_size))

test1_data = [1 -1 0; 2 2 -1]
y_ = [1 0 0]

"""
    Network(layer_dims, n_layers::Int)
layer_dims is input as a vector of integers defining how many neurons in each layer.
The first and last should match the size of intup and output to train and test.
"""
mutable struct Network
    layer_dims::Vector{Int}
    n_layers::Int = length(layer_dims)
end

"""
    Layer(W, b, cache)
W is the weights matrix, b is the biases vector, and Z is the vector of values computed.
"""
mutable struct Layer
    W::Array{Float32}
    b::Vector{Float32}
    Z::Array{Float32}
    cache::Array
end

sigmoid(x::Float32)::Float32 = 1/(1 + exp(-x))

function sigmoid_back(x::Float32)::Float32
    s = sigmoid(x)
    s*(1 - s)
end

ReLU(x::Float32)::Float32 = x > 0 ? x : 0

ReLU_back(x::Float32)::Float32 = x > 0 ? 1 : 0


function init_para(net::Network)
    para = Vector{Layer}
    for i in 2:net.n_layers
        para[i] = Layer(rand(Float32, (net.layer_dims[i], net.layer_dims[i - 1])), zeros(net.layer_dims[i]), zeros(net.layer_dims[i]), zeros(net.layer_dims[i]))
    end

    para
end

function fwd_prop(input::Array, para)
    n = length(para)
    para[1].cache = input
    for i in 2:n
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

function back_prop(T, Y, para)
    n = length(para)
    @assert length(Y) == length(T) "Not the same size"
    error = Y - T
    para[n].W = para[n].W + error*para[n - 1].cache'
    para[n].b = para[n].b + error
end

function training()
    nothing
end

function run(X, Y, dims::Vector, n_it::Int = 100)
    Net = Network(dims)
    parameters = init_para(Net)
    parameters = fwd_prop(X, parameters)

end