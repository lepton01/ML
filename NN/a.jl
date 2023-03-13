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

function fwd_prop(X::Array{Float64},
    para::Dict{String, Array{Float64}},
    aktiv::Tuple)::Tuple

    n_layers::Int = length(para) + 2
    cache = Dict{String, Array{Float64}}()
    cache[string("A", 0)] = X

    for i in 1:n_layers
        begin
            wi = para[string("w", i)]
            Ai_ant = cache[string("A", i - 1)]
            yi = zeros(Float64, (size(wi)[1], size(Ai_ant)[2]))
            mul!(yi, wi, Ai_ant)
            yi .+= para[string("b", i)]
        end
        Ai = aktiv[i].(yi)

        cache[string("y", i)] = yi
        cache[string("A", i)] = Ai
    end
    Ai, cache
end
"""
    Forward propagation.
"""

function cost_bin(A::Array{Float64},
    B::Array{Float64})::Float64

    @assert length(A) == length(B)
    cost = - sum(A.*log.(B) .+ (1 .- A).*log.(1 .- B))/length(A)
    cost
end
"""
    Cost computation.
"""

function back_prop(A::Array,
    B::Array,
    para::Array,
    cache::Array,
    layer_dims::Vector,
    aktiv::Tuple)::Dict{String, Array{Float64}}

    n_layers = length(layer_dims)
    @assert length(A) == length(B)
    m = size(S)[2]

    dA = (.-A/B .+ (1 .- A))./(1 .- B)
    if all(isnan.(dA))
        println("¡dA es NaN!")
        dA = rand(Float64)
    end
    grads = Dict{String, Array{Float64}}()
    for i in n_layers - 1:-1:1
        dy = dA.*aktiv[i].(cache[string("y", i)])
        grads[string("dw", i)] = 1/m.*(dy*transpose(caches[string("A", l-1)]))
        grads[string("db", i)] = 1/m.*sum(dy, dims = 2)
        dA = transpose(parameters[string("W", l)])*dy
    end
    grads
end

function up_para(para::Dict{String, Array{Float32}}, grads::Dict{String, Array{Float32}}, layer_dims::Array{Int}, learn_rate::Number)::Dict{String, Array{Float32}}
    n_layers = length(layer_dims)
    for i = 1:n_layers - 1
        para[string("w", i)] -= learn_rate.*grads[string("dw", i)]
        para[string("b", i)] -= learn_rate.*grads[string("db", i)]
    end
    para
end

function get_back_activations(activations)
    activations_back = []
    for activation in activations
        push!(activations_back, @eval ($(Symbol("$activation", "_back"))))
    end
    activations_back = Tuple(activations_back)
    return activations_back
end

# Prevent Rank 0 array and use Float32 for better consistency
function reshape_A(A::Array)
    A = convert(Array{Float32, ndims(A)}, A)
    A = reshape(A, 1, :)
    A
end

function main()
    layers::Int = 2
    w = rand()
    nothing
end