#01/04/2023
using LinearAlgebra, Statistics, Random
using Plots
using Flux, SpecialFunctions
using Flux: mae, crossentropy, train!
Random.seed!(1)
"""
    bessel_appx(x, a, ep)

Approximates the first kind Bessel function centered at `a`.
"""
function bessel_appx(x::Vector{Float32}, a::Float32, ep::Int = 10_000)
    Y_train = map(x) do i
        real(besselj(a, i))
    end
    X_train = vcat(x', fill(a, (1, length(x))))
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
    
    model = Chain(
        BatchNorm(2),
        Dense(2 => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => 256, relu),
        Dense(256 => 1)
    ) |> gpu
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
    loss(m, x, y) = mae(m(x), y)
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
        if rem(i, 500) == 0
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
    Y_hat = model(X_train |> gpu) |> cpu
    lab = ["Bessel" "App"]
    p = plot(x, Y_train)
    plot!(x, Y_hat', labels = lab[2])
    savefig(p, "besselj")
    return mean(isapprox.(Y_hat, Y_train'; atol = 0.01))*100
end

@time bessel_appx(collect(Float32, LinRange(0.01, 50, 2000)), Float32(0.5))
