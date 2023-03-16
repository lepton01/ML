#25/02/2023

using Plots
using LinearAlgebra, Statistics, Flux, Random, MLDatasets, BSON
using Flux: crossentropy, onecold, onehotbatch, train!

Random.seed!(1)

x_tr_raw, y_tr_raw = MNIST.(split=:train)[:]
x_te_raw, y_te_raw = MNIST.(split=:test)[:]

x_train = Flux.flatten(x_tr_raw)
x_test = Flux.flatten(x_te_raw)

y_train = onehotbatch(y_tr_raw, 0:9)
y_test = onehotbatch(y_te_raw, 0:9)


model1 = Chain(
    Dense(28^2, 2, relu),
    Dense(2, 10),
    softmax
)

model2 = Chain(
    Dense(28^2, 128, relu),
    Dense(128, 10),
    softmax
)

model3 = Chain(
    Dense(28^2, 2, relu),
    Dense(2, 4, relu),
    Dense(4, 10),
    softmax
)

model4 = Chain(
    Dense(28^2, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 10),
    softmax
)

model5 = Chain(
    Dense(28^2, 1, relu),
    Dense(1, 1, relu),
    Dense(1, 1, relu),
    Dense(1, 10),
    softmax
)

model6 = Chain(
    Dense(28^2, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 10),
    softmax
)

model7 = Chain(
    Dense(28^2, 64, relu),
    Dense(64, 128, relu),
    Dense(128, 64, relu),
    Dense(64, 128, relu),
    Dense(128, 10),
    softmax
)

"""
    main(model, epoch)

executes training and testing of the network and its parameters.
model can be selected and it specifies the structure of the network, number of layers and number of neurons in each layer.
epoc is an integer representing the number of iterations to be made.
"""
function main(model, epoc::Int)
    loss(x, y) = crossentropy(model(x), y)

    ps = Flux.params(model)

    lr = 0.01
    opt = ADAM(lr)

    loss_h = []

    for i in 1:epoc
        train!(loss, ps, [(x_train, y_train)], opt)
        train_l = loss(x_train, y_train)
        push!(loss_h, train_l)
        #println("Epoch = $i : Training loss = $train_l")
    end

    y_hat_raw = model(x_test)
    y_hat = onecold(y_hat_raw) .- 1

    BSON.@save "mymodel.bson" model

    y = y_te_raw
    mean(y_hat .== y)*100
end