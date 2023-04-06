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
        Dense(2 => 16, celu),
        Dense(16 => 32),
        Dense(32 => 64),
        Dense(64 => 16),
        Dense(16 => 1))

    #loss(m, x, y) = mse(m(x), y)
    loss(m, x, y) = mae(m(x), y)
    #lr::Float32 = 0.01
    opt = Flux.setup(Flux.Adam(), model)
    loss_log = Float32[]
    for i ∈ 1:ep
        
        train!(loss, model, train_SET, opt)
        train_loss = loss(model, X_train, Y_train')
        push!(loss_log, train_loss)
        if rem(i, 100) == 0
            println("Epoch = $i. Training loss = $train_loss")
        end
        
        #=
        losses = Float32[]
        for (i, data) ∈ enumerate(train_SET)
            input, label = data

            val, grads = Flux.withgradient(model) do m
                # Any code inside here is differentiated.
                # Evaluation of the model and loss must be inside!
                result = m(input)
                mse(result, label)
            end
            # Save the loss from the forward pass. (Done outside of gradient.)
            push!(losses, val)
            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            #=
            if !isfinite(val)
                @warn "loss is $val on item $i." i
                continue
            end
            =#
            Flux.update!(opt, model, grads[1])
        end
        train_loss = loss(model, X_train, Y_train')
        if i%100 == 0
            println("Epoch = $i. Training loss = $train_loss")
        end
        # Compute some accuracy, and save details as a NamedTuple
        acc = mean(model(X_train) .≈ Y_train')
        #push!(loss_log, losses)

        # Stop training when some criterion is reached
        if acc > 0.95
            println("stopping after $epoch epochs.")
            break
        end
        =#
    end
    Y_hat = model(X_train)
    p = plot(x, Y_train)
    plot!(x, Y_hat')
    savefig(p, "besselj")
    return mean(isapprox.(Y_hat, Y_train'; atol = 0.05))*100
end

@time main(collect(Float32, LinRange(0.01, 25, 5000)), Float32(1.5))
