#01/04/2023
"""
    bessel_model_creation(x, a, model_name, ep)

Creation of a NN models
"""
function bessel_model_creation(x::Vector{Float32}, a::Vector{Float32}, model_name::String, ep::Int = 10_000)
    @assert x isa Vector "x must be of type Vector for training"
    @assert a isa Vector "a must be of type Vector for training"

    model = Chain(
        BatchNorm(2),
        Dense(2 => 1024, celu),
        Dense(1024 => 1024, celu),
        Dense(1024 => 1024, celu),
        Dense(1024 => 1024, celu),
        Dense(1024 => 1024, celu),
        Dense(1024 => 1)
    ) |> gpu
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
