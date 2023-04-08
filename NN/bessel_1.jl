#01/04/2023
using LinearAlgebra, Statistics, Random
using Plots
using Flux, SpecialFunctions, BSON
using Flux: mae, train!
#Random.seed!(1)
#gr(1600, 900)
"""
    bessel_appx(x, a, ep)

Approximates the first kind Bessel function centered at `a`.
"""
function bessel_appx(x::Vector{Float32}, a::Vector{Float32}, ep::Int = 50_000)
    #=
    Y_train = map(x, a) do i, j
        besselj(j, i) |> real
    end
    X_train = vcat(x', fill(a, (1, length(x))))
    =#
    
    Y_train = map(x, a) do i, j
        besselj(j, i) |> real
    end
    X_train = vcat(x', a')
    
    train_SET = [(X_train, Y_train')] |> gpu
    #=
    model = Chain(
        Dense(2 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 32), relu,
        Dense(32 => 1)
    ) |> gpu
    =#
    #=)
    model = Chain(
        BatchNorm(2),
        Dense(2 => 256, relu),
        Dense(256 => 256, relu),
        Dense(256 => 256, relu),
        Dense(256 => 1)
    ) |> gpu
    =#
    #=
    model = Chain(
        Dense(2 => 16, relu),
        Dense(16 => 32, relu),
        Dense(32 => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => 16, relu),
        Dense(16 => 1)
    ) |> gpu
    model = Chain(
        Dense(2 => 16, relu),
        Dense(16 => 32, relu),
        Dense(32 => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => 16, relu),
        Dense(16 => 1)
    ) |> gpu
    =#
    BSON.@load "bessel_j.bson" model
    #loss(m, x, y) = mae(m(x), y)
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i ∈ 1:ep
        #=
        train!(loss, model, train_SET, opt)
        train_loss = loss(model, X_train, Y_train')
        push!(loss_log, train_loss)
        if rem(i, 100) == 0
            println("Epoch = $i. Training loss = $train_loss")
        end
        =#

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
    x_test = 50*rand32(500)
    a_test = 50*rand32(500)
    X_test = vcat(x_test', a_test')
    Y_test = map(x, a) do i, j
        besselj(j, i) |> real
    end
    Y_hat = model(X_test |> gpu) |> cpu
    model |> cpu
    BSON.@save "bessel_j.bson" model
    return mean(isapprox.(Y_hat', Y_test; atol = 0.01))*100
end

@time bessel_appx(collect(Float32, LinRange(0.01, 50, 500)), 50*rand32(500))
