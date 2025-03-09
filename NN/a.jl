#Angel Ortiz
#11/03/2023
using Statistics, Random, LinearAlgebra, MLJBase
"""
    Sigmiod activation function.
"""
sigmoid(x::Real) = one(x) / (one(x) + exp(-x))

function sigmoid_back(x::Real)
    s = sigmoid(x)
    return s * (one(s) - s)
end
"""
    Rectified Linear Unit activation function.
"""
ReLU(x::Real) = x > zero(x) ? x : zero(x)

ReLU_back(x::Real) = x > zero(x) ? one(x) : zero(x)
"""
    Initialization of weights (wi) and biases (bi). Each layer is represented by a dictionary with keys being the names for the w and b of each neuron i.
"""
function init_para(layer_dims::Vector)::Dict{String,Array{Float64}}
    A = Dict{String,Array{Float64}}()
    for ii ∈ 2:eachindex(layer_dims)
        A[string("w", ii - 1)] = rand((layer_dims[ii], layer_dims[ii-1])) / sqrt(ii - 1)
        A[string("b", ii - 1)] = zeros(layer_dims[ii])
    end
    return A
end
"""
    Forward propagation.
"""
function fwd_prop(X::Array{Float64},
    para::Dict{String,Array{Float64}},
    aktiv::Tuple)
    n_layers::Int = length(para) ÷ 2
    cache = Dict{String,Array{Float64}}()
    cache[string("A", 0)] = X
    for ii ∈ 1:n_layers
        begin
            wi = para[string("w", ii)]
            Ai_ant = cache[string("A", ii - 1)]
            yi = zeros(Float64, (size(wi)[1], size(Ai_ant)[2]))
            mul!(yi, wi, Ai_ant)
            yi .+= para[string("b", ii)]
        end
        Ai = aktiv[ii].(yi)
        cache[string("y", ii)] = yi
        cache[string("A", ii)] = Ai
    end
    return Ai, cache
end
"""
    Cost computation.
"""
function cost_bin(A::Array{Float64},
    B::Array{Float64})::Float64
    @assert length(A) == length(B) "Not the same size."
    return -sum(A .* log.(B) .+ (1 .- A) .* log.(1 .- B)) / length(A)
end
"""
    Calculate changes in the parameters.
"""
function back_prop(A::Array,
    B::Array,
    para::Dict,
    cache::Dict,
    layer_dims::Vector,
    aktiv::Tuple)::Dict{String,Array{Float64}}
    n_layers = length(layer_dims)
    @assert length(A) == length(B) "Not the same size."
    l = size(A, 2)
    dA = -A ./ B .+ (1 .- A) ./ (1 .- B)
    if all(isnan.(dA))
        println("¡dA es NaN!")
        dA = rand()
    end
    grads = Dict{String,Array{Float64}}()
    for ii ∈ n_layers-1:-1:1
        dy = dA .* aktiv[i].(cache[string("y", ii)])
        grads[string("dw", ii)] = 1 / l .* (dy * transpose(cache[string("A", ii - 1)]))
        grads[string("db", ii)] = 1 / l .* sum(dy, dims=2)
        dA = transpose(para[string("w", ii)]) * dy
    end
    return grads
end
"""
    Update parameters.
"""
function up_para(para::Dict{String,Array{Float64}},
    grads::Dict{String,Array{Float64}},
    layer_dims::Array{Int},
    learn_rate::Number)::Dict{String,Array{Float64}}
    n_layers = length(layer_dims)
    for ii ∈ 1:n_layers-1
        para[string("w", ii)] -= learn_rate .* grads[string("dw", ii)]
        para[string("b", ii)] -= learn_rate .* grads[string("db", ii)]
    end
    return para
end
"""
    Obtain the activations.
"""
function get_aktiv(aktiv)
    aktiv_back = []
    for activation ∈ aktiv
        push!(aktiv_back, @eval($(Symbol("$activation", "_back"))))
    end
    return Tuple(aktiv_back)
end
"""
    ...
"""
function reshape_M!(M::Array)
    M = convert(Array{Float64,ndims(M)}, M)
    return reshape(M, 1, :)
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
function neural_net_dense(X::Vector, Y::Vector,
    layer_dims::Array{Int},
    n_it::Int,
    learn_rate::Real;
    activations=nothing,
    print_stats=false,
    parameters=nothing,
    resume=false,
    checkpoint_steps=100)
    n_layers = length(layer_dims)
    Y = reshape_M!(Y)
    @assert ndims(Y) == 2 "No dimension match."
    begin
        if activations === nothing
            activations = Vector{Function}(undef, n_layers - 1)
            for ii ∈ 1:n_layers-2
                activations[ii] = ReLU
            end
            activations[n_layers-1] = sigmoid
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
        for ii ∈ eachindex(parameters)
            println("\tInitial mean of parameter $ii is ", mean(parameters[ii]), ".")
            println("\tInitial variance of parameter $ii is ", var(parameters[ii]), ".")
        end
    end
    begin
        for iteration ∈ 1:n_it
            B, cache = fwd_prop(X, parameters, activations)
            grads = back_prop(Y, B, parameters, cache, layer_dims, activations_back)
            parameters = up_para(parameters, grads, layer_dims, learn_rate)
            begin
                if iteration % checkpoint_steps == 0
                    cost = cost_bin(Y, B)
                    println("Cost at iteration $iteration is $cost")
                    if print_stats
                        for ii in eachindex(parameters)
                            println("\tMean of parameter $ii is ", mean(parameters[ii]))
                            println("\tVariance of parameter $ii is ", var(parameters[ii]))
                        end
                    end
                end
            end
        end
    end
    return parameters, activations
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
function predict(X, Y, para::Array, activations::Tuple)
    m = size(X)[2]
    #l = length(para)
    predicts = zeros((1, m))
    #copy Y to CPU
    if Y !== nothing
        Y = Array(Y)
        Y = reshape_Y(Y)
    end
    probs, _ = fwd_prop(X, para, activations)
    begin
        probs = Array(probs)
        for ii = 1:m
            probs[1, ii] > 0.5 ? predicts[1, ii] = 1 : predicts[1, ii] = 0
        end
    end
    accuracy = nothing
    if Y !== nothing
        accuracy = sum(predicts .== Y) / m
        println("Accuracy is $(accuracy*100)%.")
    end
    return predicts, accuracy
end
x_tr_raw, y_tr_raw = MNIST.(split=:train)[:]
x_te_raw, y_te_raw = MNIST.(split=:test)[:]
