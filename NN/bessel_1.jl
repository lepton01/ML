#01/04/2023

using LinearAlgebra, Statistics, Random
using Plots
using Flux, SpecialFunctions
using Flux: mse, train!

Random.seed!(1)
"""
    main()

¿? SOMETHING
"""
function main(x::Vector{Float32}, z::Float32, ep::Int = 1_000)
    target = map(x) do i
        real(besselj(z, i))
    end
    
    p = plot(x, target)
    
    X = vcat(x', zeros(Float32, length(x))')
    X[2, :] .= z
    loader = Flux.DataLoader((X, target), batchsize = 64, shuffle = true)

    model = Chain(
        Dense(2 => 2, relu),
        Dense(2 => 4, relu),
        Dense(4 => 2, relu),
        Dense(2 => 1),
    softmax)

    loss(m, x, y) = mse(m(x), y)
    lr::Float32 = 0.01
    opt_st = Flux.setup(Adam(), model)
    loss_h = Float32[]
    for i ∈ 1:ep
        train!(loss, model, [(X, target')], opt_st)
        train_l = loss(model, X, target')
        push!(loss_h, train_l)
        println("Epoch = $i. Training loss = $train_l")
    end
    ŷ = model(X)
    p = plot!(x, ŷ')
    savefig(p, "besselj")
    return mean(ŷ .== target')*100
end

main(collect(Float32, LinRange(0.01, 10, 1000)), Float32(1.5))