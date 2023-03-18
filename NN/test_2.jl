#17/03/2023

using LinearAlgebra, Statistics, Random
using Plots

#include("datagen.jl")

mutable struct Network
    layer_dims::Vector{Int}
    n_layers::Int
end

mutable struct Layer
    W::Array{Float32}
    b::Vector{Float32}
    a_ant::Vector{Float32}
    a::Vector{Float32}
end

sigmoid(x::Float64)::Float64 = 1/(1 + exp(-x))

function sigmoid_back(x::Float64)::Float64
    s = sigmoid(x)
    s*(1 - s)
end

ReLU(x::Float64)::Float64 = x > 0 ? x : 0

ReLU_back(x::Float64)::Float64 = x > 0 ? 1 : 0


function init_para(net::Network)
    for i in 1:net.n_layers - 1
        
    end
    
    para
end

function training()
    nothing
end