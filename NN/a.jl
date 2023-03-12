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

function fwd_prop(X::Array{Float64}, para::Dict{String, Array{Float64}}, activ::Tuple)::Tuple
    n_layers::Int = length(para) + 2
    cache = Dict{String, Array{Float64}}()
    cache[string("a", 0)] = X

    for i in 1:n_layers
        begin
            wi = para[string("w", i)]
            ai_ant = cache[string("a", i - 1)]
            yi = zeros(Float64, (size(wi)[1], size(ai)[2]))
            mul!(yi, wi, ai_ant)
            yi .+= para[string("b", i)]
        end
        ai = activ[i].(yi)

        cache[string("y", i)] = yi
        cache[string("a", i)] = ai
    end
    ai, cache
end

function cost()
    
end

function main()
    layers::Int = 2
    w = rand()
    nothing
end