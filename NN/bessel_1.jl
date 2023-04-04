#01/04/2023

using LinearAlgebra, Statistics, Random
using Plots
using Flux, SpecialFunctions
using Flux: crossentropy, train!, params

Random.seed!(1)
"""
    main()

¿? SOMETHING
"""
function main(x::Vector{Float32}, z::Float32, ep::Int = 1000)
    y = map(x) do a
        real(besselj(z, a))
    end
    p = plot(x, y)
    
    X = vcat(x', zeros(Float32, length(x))')
    X[2, :] .= z
    model = Chain(
        Dense(2, 8, relu),
        Dense(8, 16, relu),
        Dense(16, 8, relu),
        Dense(8, 1),
    softmax)

    loss(x, y) = crossentropy(model(x), y)
    para = params(model)
    lr::Float32 = 0.01
    opt = ADAM(lr)
    loss_h = Float32[]
    for i ∈ 1:ep
        train!(loss, para, [(X, y')], opt)
        train_l = loss(X, y')
        push!(loss_h, train_l)
        println("Epoch = $i. Training loss = $train_l")
    end
    ŷ = model(X)
    p = plot!(x, ŷ')
    savefig(p, "besselj")

    return mean(ŷ .== y')*100
end

main(collect(Float32, LinRange(0.01, 10, 1000)), Float32(1.5))