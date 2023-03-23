#17/03/2023

using LinearAlgebra, Statistics, Random
using Plots

#include("datagen.jl")

#X = Float64.(hcat(real, fake))
#Y = vcat(ones(train_size), zeros(train_size))

x_ = [1 1 2 2 -1 -2 -1 -2; 1 2 -1 0 2 1 -1 -2]
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
    W::Array{Float64}
    b::Vector{Float64}
    Z::Array{Float64}
    cache::Array{Float64}
end

"""
    sigmoid(x::Float64)

Sigmoid (`σ`) activation function.\\
Returns a ``Float64 number``.
"""
sigmoid(x::Float64)::Float64 = 1/(1 + exp(-x))

"""
    sigmoid_der(x::Float64)

'Derivative' sigmoid (`σ`) activation function.\\
Returns a `Float64` number.
"""
function sigmoid_der(x::Float64)::Float64
    s = sigmoid(x)
    s*(1 - s)
end

"""
    ReLU(x::Float64)

Rectified linear unit activation function. Same effect as ``max(0, x)``.\\
Returns a `Float64` number.
"""
ReLU(x::Float64)::Float64 = x > 0 ? x : 0

"""
    ReLU_der(x::Float64)

'Derivative' rectified linear unit activation function.\\
Returns a `Float64` number.
"""
ReLU_der(x::Float64)::Float64 = x > 0 ? 1 : 0

"""
    init_para(net)

Creates a vector storing type ``Layer`` with the parameters (weights, biases, and values) for all neurons in every layer.
"""
function init_para(net::Network)::Array
    para = Layer[]
    for i ∈ 1:length(net.layer_dims)
        i == 1 ? push!(para, Layer(zeros(Float64, (1, 1)), zeros(Float64, net.layer_dims[i]), zeros(Float64, net.layer_dims[i]), zeros(Float64, net.layer_dims[i]))) : push!(para, Layer(rand(Float64, (net.layer_dims[i], net.layer_dims[i - 1])), zeros(Float64, net.layer_dims[i]), zeros(Float64, net.layer_dims[i]), zeros(Float64, net.layer_dims[i])))
    end

    para
end

"""
    fwd_prop!(input::Array, para)

Modifies the ``para`` argument. ``input`` is an array of the initial inputs to the network.\\
``para`` is a ``Vector{Layer}`` containing all the parameters.
"""
function fwd_prop!(input::Array, para::Array)::Array
    n = length(para)
    para[1].cache = input
    for i ∈ 2:n
        para[i].Z = muladd(para[i].W, para[i - 1].cache, para[i].b)
        i < n ? para[i].cache = ReLU.(para[i].Z) : para[i].cache = sigmoid.(para[i].Z)
        #para[i].cache = sigmoid.(para[i].Z)
    end

    para
end

"""
    cost(T, Y)

a
"""
function cost(T::Vector, Y::Vector)::Float64
    @assert length(Y) == length(T) "Not the same size."
    l = length(Y)

    cost = -sum(Y.*log.(T) .+ (1 .- Y).*log.(1 .- T))/l
end

"""
    back_prop!(T, Y, para)

Modifies ``para`` input. Compares between output and expected output and changes the weights and biases depending on the cost function.
"""
function back_prop!(T, Y, para, l_rate)
    n = length(para)
    @assert length(Y) == length(T) "Not the same size"
    error = (Y - T)
    errorsqrd = (error.^2)./length(Y)
    g1 = -2*errorsqrd
    #g2 = sigmoid_der(T)
    for i ∈ 1:lastindex(para) - 1
        i != 1 ? g2 = ReLU_der.(T) : g2 = sigmoid_der.(T)
        if i == lastindex(para) - 1
            nothing
        else
            para[end - i + 1].W -= l_rate*g2*para[end - i].cache'
            para[end - i + 1].b -= l_rate*g2
        end

    end

    para
end


"""
    training(X, Y, dims, epochs)

``X`` is the input array, ``Y`` is the expected output, ``dims`` is a vector specifying the number of neurons in each layer,\\
``learn_r`` is the learning rate of training, can be specified by user, and ``epochs`` is the number of iterations for training, can be specified by user.\\
Calls creation and propagation functions, modifies the parameters for layers,\\
and returns the final ``parameters`` array containing the info for all layers and their neurons.
"""
function training(X, Y, dims::Vector, epochs::Int = 100)
    r(t::Int) = exp(-t)

    Net = Network(dims)
    parameters = init_para(Net)
    if size(X, 1) != 2
        reshape!(X, 2, :)
    end
    if size(Y, 1) != 2
        reshape!(Y, 2, :)
    end
    for i ∈ 1:epochs
        for j ∈ 1:length(X[1, :])
            parameters = fwd_prop!(X[:, j], parameters)
            #c = cost(parameters[end].cache, Y[:, j])
            parameters = back_prop!(parameters[end].cache, Y[:, j], parameters, r(i))
        end
        
        i != epochs ? nothing : show(parameters[end].cache)
    end

    parameters
end

"""
    testing()

a
"""
function testing()
    nothing
end

paras = training(x_, y_, [2, 2, 2], 10000)