using LinearAlgebra, Statistics
using Plots
using Flux, SpecialFunctions
using Flux: train!, mse

#Random.seed!(1)

x = collect(Float32, LinRange(0.01, 10, 1000))
a = Float32(0.5)
#=
y_train = map(x) do i
    real(besselj(a, i))
end
p = plot(x, y_train)
savefig(p, "besselj")
=#

y_train = map(x) do i
    i^2*sin(a*i)
end
p = plot(x, y_train)
savefig(p, "cosa(ax)")

X_train = vcat(x', fill(a, (1, length(x))))
#X = Flux.flatten(X)
#loader = Flux.DataLoader((X, target), batchsize = 64, shuffle = true)
train_SET = [(X_train, y_train')]

model = Chain(
    Dense(2 => 2, relu),
    Dense(2 => 1))

loss(m, x, y) = mse(m(x), y)
opt = Descent()
model(X_train)

train!(loss, model, train_SET, opt)
for data ∈ train_SET
    input, label = data

    grads = Flux.gradient(model) do m
        result = m(input)
        loss(m, result, label)
    end
    Flux.update!(opt_st, model, grads[1])
end
