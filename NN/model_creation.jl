#01/04/2023
using LinearAlgebra, Statistics, Random
using Plots
using Flux, SpecialFunctions, BSON
using Flux: mae
#Random.seed!(1)
#gr(1600, 900)
"""
    bessel_appx(x, a, ep)

Approximates the first kind Bessel function centered at `a`.
"""
function bessel_model_creation(x::Vector{Float32}, a::Vector{Float32}, model_name::String, ep::Int = 10_000)
    @assert x isa Vector "x must be of type Vector for training"
    @assert a isa Vector "a must be of type Vector for training"

    Y_train = map(x, a) do i, j
        besselj(j, i) |> real .|> Float32
    end
    X_train = vcat(x', a')
    train_SET = [(X_train, Y_train')] |> gpu
    model = Chain(
        BatchNorm(2),
        Dense(2 => 64, relu),
        Dense(64 => 64, relu),
        Dense(64 => 1, relu)
    ) |> gpu
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i ∈ 1:ep
        losses = Float32[]
        for data ∈ train_SET
            input, label = data
        
            l, grads = Flux.withgradient(model) do m
                result = m(input)
                mae(result, label)
            end
            push!(losses, l)
            Flux.update!(opt, model, grads[1])
        end
        l2 = sum(losses)
        push!(loss_log, l2)
        if rem(i, 1000) == 0
            println("Epoch = $i. Training loss = $l2")
        end
        #=
        # Stop training when some criterion is reached
        acc = mean(isapprox.(model(X_train), Y_train'; atol = 0.05))
        if acc > 0.95
            println("stopping after $epoch epochs.")
            break
        end
        =#
    end
    
    #lab = ["Bessel" "App"]
    #p = plot(x, Y_train, labels = lab[1])
    #plot!(x, Y_hat', labels = lab[2])
    #savefig(p, "besselj")
    x_test = 50*rand32(length(x))
    a_test = 50*rand32(length(a))
    X_test = vcat(x_test', a_test')
    Y_test = map(x_test, a_test) do i, j
        besselj(j, i) |> real
    end
    Y_hat = model(X_test |> gpu) |> cpu
    model = model |> cpu
    BSON.@save model_name model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.02))*100
end
#@time bessel_model_creation(collect(Float32, LinRange(0.01, 50, 5000)), 50*rand32(5000), "bessel_j_.bson")
