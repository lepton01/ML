#11/03/2023

using Plots
using Statistics, Random, LinearAlgebra, MLJBase, MLDatasets

#include("datagen.jl")


X = Float32.(hcat(real, fake))
Y = vcat(ones(train_size), zeros(train_size))

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
    Initialization of weights (wi) and biases (bi). Each layer is represented by a dictionary with keys being the names for the w and b of each neuron i.
"""
function init_para(layer_dims::Vector)::Dict{String, Array{Float32}}
    A = Dict{String, Array{Float32}}()
    for i in 2:length(layer_dims)
        A[string("w", i - 1)] = rand(Float32, (layer_dims[i], layer_dims[i - 1]))/sqrt(i - 1)
        A[string("b", i - 1)] = zeros(Float32, layer_dims[i])
    end

    A
end


"""
    Forward propagation.
"""
function fwd_prop(X::Array{Float32},
    para::Dict{String, Array{Float32}},
    aktiv::Tuple)

    n_layers::Int = length(para) ÷ 2
    cache = Dict{String, Array{Float32}}()
    cache[string("A", 0)] = X

    Ai = nothing
    for i in 1:n_layers
        begin
            wi = para[string("w", i)]
            Ai_ant = cache[string("A", i - 1)]
            yi = zeros(Float32, (size(wi)[1], size(Ai_ant)[2]))
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
    Cost computation.
"""
function cost_bin(Y::Array{Float32},
    T::Array{Float32})::Float32

    @assert length(Y) == length(T) "Not the same size."
    l = length(Y)
    cost = -sum(Y.*log.(T) .+ (1 .- Y).*log.(1 .- T))/l
end


"""
    Calculate changes in the parameters.
"""
function back_prop(A::Array,
    B::Array,
    para::Dict,
    cache::Dict,
    layer_dims::Vector,
    aktiv::Tuple)::Dict{String, Array{Float32}}

    n_layers = length(layer_dims)
    @assert length(A) == length(B) "Not the same size."
    l = size(A)[2]
    dA = -A./B .+ (1 .- A)./(1 .- B)
    if all(isnan.(dA))
        println("¡dA es NaN!")
        dA = rand(Float32)
    end

    grads = Dict{String, Array{Float32}}()
    for i in n_layers - 1:-1:1
        dy = dA.*aktiv[i].(cache[string("y", i)])
        grads[string("dw", i)] = 1/l.*(dy*transpose(cache[string("A", i - 1)]))
        grads[string("db", i)] = 1/l.*sum(dy, dims = 2)
        dA = transpose(para[string("w", i)])*dy
    end

    grads
end


"""
    Update parameters.
"""
function up_para(para::Dict{String, Array{Float32}},
    grads::Dict{String, Array{Float32}},
    layer_dims::Array{Int},
    learn_rate::Number)

    n_layers = length(layer_dims)
    for i = 1:n_layers - 1
        para[string("w", i)] -= learn_rate.*grads[string("dw", i)]
        para[string("b", i)] -= learn_rate.*grads[string("db", i)]
    end

    para
end


"""
    Obtain the activations.
"""
function get_aktiv(aktiv)
    aktiv_back = []
    for activation in aktiv
        push!(aktiv_back, @eval($(Symbol("$activation", "_back"))))
    end

    aktiv_back = Tuple(aktiv_back)
end


"""
    ...
"""
function reshape_M!(M::Array)
    M = convert(Array{Float32, ndims(M)}, M)
    M = reshape(M, 1, :)
end


"""
    neural_network_dense(X, Y, layer_dims::Array{Int}, num_iterations::Int, learning_rate::Number; activations=Nothing, print_stats=false, parameters=nothing, resume=false, checkpoint_steps=100)
Build and train a dense neural network for binary classification. Performs batch gradient descent for optimization and uses binary logistic loss for cost function.
Also supports resuming training from previously trained parameters.

Parameters:
- `X`: training inputs (the first value in size of this should be the same as the first value in layer_dims)
- `Y`: training outputs (for binary classification, first value of size should be 1)
- `layer_dims`: Number of neurons in each layer. First layer size should be equal to the number of features in input. Output layer should be 1 in case of binary classification
- `num_iterations`: Number of iterations to train for.
- `learning_rate`: for gradient descent optimizer
- `activations` (optional): A tuple of activation functions for every layer other than the input layer (default: (relu, relu...., sigmoid))
- `print_stats` (optional): Whether to print mean and variance of parameters for every `checkpoint_steps` steps (Statistics package must be imported to use this) (default: false)
- `checkpoint_steps` (optional): the loss (and stats, if print_stats is true) is printed every `checkpoint_steps` steps (default: 100)
- `resume` (optional): whether to resume training by using the given parameters (if parameters are not given, they are initialized again) (default: false)
- `parameters` (optional): parameters for resuming training from checkpoint. Used only if resume is true (default: nothing)

Returns:
- `parameters`: trained parameters
- `activations`: activations used in training (for passing them to `predict` function)
"""
function neural_net_dense(X, Y,
    layer_dims::Array{Int},
    n_it::Int,
    learn_rate::Real;
    activations = nothing,
    print_stats = false,
    parameters = nothing,
    resume = false,
    checkpoint_steps = 100)

    n_layers = length(layer_dims)
    Y = reshape_M!(Y)
    @assert ndims(Y) == 2 "No dimension match."

    begin
        if activations === nothing
            activations = Vector{Function}(undef, n_layers - 1)
            for i = 1:n_layers - 2
                activations[i] = ReLU
            end

            activations[n_layers - 1] = sigmoid
        end

        activations = Tuple(activations)
        activations_back = get_aktiv(activations)
    end

    begin
        init_params = false
        if !resume
            init_params = true
        elseif resume && parameters === nothing
            println("Cannot resume without parameters, pass parameters = parameters to resume training, reinitializing parameters.")
            init_params = true
        end

        if init_params
            parameters = init_para(layer_dims)
        end
    end

    if print_stats
        for i in eachindex(parameters)
            println("\tInitial mean of parameter ", i, " is ", mean(parameters[i]), ".")
            println("\tInitial variance of parameter ", i, " is ", var(parameters[i]), ".")
        end
    end

    begin
        for iteration = 1:n_it
            T, cache = fwd_prop(X, parameters, activations)
            grads = back_prop(Y, T, parameters, cache, layer_dims, activations_back)
            parameters = up_para(parameters, grads, layer_dims, learn_rate)

            begin
                if iteration % checkpoint_steps == 0
                    cost = cost_bin(Y, T)
                    println("Cost at iteration $(iteration) is $(cost)")
                    if print_stats
                        for i in eachindex(parameters)
                            println("\tMean of parameter ", i, " is ", mean(parameters[i]))
                            println("\tVariance of parameter ", i, " is ", var(parameters[i]))
                        end
                    end
                end
            end
        end
    end

    parameters, activations
end


"""
    predict(X, Y, parameters, activations::Tuple)
Predict using the trained parameters and calculate the accuracy.

Parameters:
- `X`: testing inputs
- `Y`: outputs to check with
- `parameters`: trained parameters which is taken from the output of `neural_network_dense`
- `activations`: also given by the outputs of `neural_network_dense`

Returns:
- `predicts`: predictions from the NN
- `accuracy`: accuracy of predictions
"""
function predict(X, Y, para, activations::Tuple)
    m = size(X)[2]
    l = length(para)
    predicts = zeros((1, m))

    # Copy Y to CPU
    if Y !== nothing
        Y = Array(Y)
        Y = reshape_M!(Y)
    end

    probs, cache = fwd_prop(X, para, activations)
    begin
        probs = Array(probs)
        for i = 1:m
            probs[1, i] > 0.5 ? predicts[1, i] = 1 : predicts[1, i] = 0
        end
    end

    accuracy = nothing
    if Y !== nothing
        accuracy = sum(predicts .== Y)/m
        println("Accuracy is $(accuracy*100)%.")
    end

    predicts, accuracy
end

para, act = neural_net_dense(X, Y, [2, 16, 16, 1], 100, 0.01)

predict(X, Y, para, act)