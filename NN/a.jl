#11/03/2023

using Plots
using Statistics, Random, LinearAlgebra
using MLJBase, StableRNGs
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