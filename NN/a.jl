#11/03/2023

using Plots
using Statistics, Random, LinearAlgebra
using MLJBase
"""
    Packages and dependencies.
"""


function ReLU(A)
    a = max.(0, A)
    (A, a)
end
"""
    ReLU activation function.
"""


function init_params(layer_dims::Vector)
    A = Dict{String, Array{Float64}}()
    for i in 2:eachindex(layer_dims)
        A[string("w", i - 1)] = rand(Float64, (layer_dims[i], layer_dims[i - 1]))/sqrt(i - 1)
        A[string("b", i - 1)] = zeros(Float64, layer_dims[i])
    end
    A
end
"""
    Initialization of weights and biases.
"""

function fwd_prop(x::Array{Float64}, )
    
end

function main()
    layers::Int = 2
    w = rand()
    nothing
end