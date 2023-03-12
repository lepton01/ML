#11/03/2023

using Plots
using Statistics, Random, LinearAlgebra
using Images
"""
    Packages and dependencies.
"""

mutable struct params
    w::Float64
    bias::Float64
end

function ReLU(A)
    a = max.(0, A)
    (A, a)
end

function input()

    nothing
end

function init_params(layer_dims::Tuple, seed)
    A = Array{params}(undef, layer_dims)
    for i in 2:length(layer_dims)
        A[i, ].w
    end
end

function main()
    layers::Int = 2
    w = rand()
    nothing
end