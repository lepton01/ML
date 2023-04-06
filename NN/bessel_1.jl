#01/04/2023

using LinearAlgebra, Statistics, Random
using Plots
using Flux, SpecialFunctions
using Flux: mse, crossentropy, train!

Random.seed!(1)
"""
    main()

¿? SOMETHING
"""
function main(x::Vector{Float32}, a::Float32, ep::Int = 10_000)
    y_train = map(x) do i
        real(besselj(a, i))
    end
    p = plot(x, y_train)
    savefig(p, "besselj")
    
    X_train = vcat(x', fill(a, (1,length(x))))
    train_set = [(X_train, y_train')]

    model = Chain(
        Dense(2 => 2, relu),
        Dense(2 => 1))

    loss(m, x, y) = mse(m(x), y)
    #lr::Float32 = 0.01
    opt_st = Flux.setup(Adam(), model)
    #opt = Descent()
    loss_log = Float32[]
    for i ∈ 1:ep
        train!(loss, model, train_set, opt_st)
        train_loss = loss(model, X_train, y_train')
        push!(loss_log, train_loss)
        println("Epoch = $i. Training loss = $train_loss")
        
        #=
        losses = Float32[]
        for (i, data) ∈ enumerate(train_set)
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
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end

            Flux.update!(opt_st, model, grads[1])
        end

        # Compute some accuracy, and save details as a NamedTuple
        acc = mean(model(X) .== target')
        #push!(loss_log, (; acc, losses))

        # Stop training when some criterion is reached
        if  acc > 0.95
            println("stopping after $epoch epochs")
            break
        end
        #push!(loss_log, losses)
        =#
    end
    y_hat = model(X_train)
    return mean(y_hat .== y_train')*100
end

main(collect(Float32, LinRange(0.01, 50, 10000)), Float32(1.5))
