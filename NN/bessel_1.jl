#01/04/2023

using LinearAlgebra, Statistics, Random
using Plots
using Flux, SpecialFunctions
using Flux: mae, crossentropy, train!

Random.seed!(1)
"""
    main()

¿? SOMETHING
"""
function main(x::Vector{Float32}, a::Float32, ep::Int = 10_000)
    Y_train = map(x) do i
        real(besselj(a, i))
    end
    X_train = vcat(x', fill(a, (1, length(x))))
    train_SET = [(X_train, Y_train')]
    model = Chain(
        Dense(2 => 16, relu),
        Dense(16 => 32, relu),
        Dense(32 => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => 16, relu),
        Dense(16 => 1)
    )
    #=
    model = Chain(
        Dense(2 => 16, relu),
        Dense(16 => 32, relu),
        Dense(32 => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => 16, relu),
        Dense(16 => 1)
    )
    model = Chain(
        Dense(2 => 16, relu),
        Dense(16 => 32, relu),
        Dense(32 => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => 16, relu),
        Dense(16 => 1)
    )
    model = Chain(
        Dense(2 => 16, relu),
        Dense(16 => 32, relu),
        Dense(32 => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => 16, relu),
        Dense(16 => 1)
    )
    =#
    loss(m, x, y) = mae(m(x), y)
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i ∈ 1:ep        
        train!(loss, model, train_SET, opt)
        train_loss = loss(model, X_train, Y_train')
        push!(loss_log, train_loss)
        if rem(i, 100) == 0
            println("Epoch = $i. Training loss = $train_loss")
        end
        # Stop training when some criterion is reached
        acc = mean(isapprox.(model(X_train), Y_train'; atol = 0.05))
        if acc > 0.95
            println("stopping after $epoch epochs.")
            break
        end
    end
    Y_hat = model(X_train)
    p = plot(x, Y_train)
    plot!(x, Y_hat')
    savefig(p, "besselj")
    return mean(isapprox.(Y_hat, Y_train'; atol = 0.05))*100
end

@time main(collect(Float32, LinRange(0.01, 25, 5000)), Float32(1.5))
