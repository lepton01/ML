using Flux, Statistics
using Flux: train!, mse


actual(x) = 4x + 2
x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)
train_set = [(x_train, y_train)]

model = Dense(1 => 1)
model(x_train)

loss(m, x, y) = mse(m(x), y)
loss(model, x_train, y_train)

opt = Descent()

train!(loss, model, train_set, opt)
loss(model, x_train, y_train)

for i in 1:100
    train!(loss, model, train_set, opt)
end
loss(model, x_train, y_train)