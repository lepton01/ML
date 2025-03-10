# Install everything, including CUDA, and load packages:
using Flux, Statistics
# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}
# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Chain(
    Dense(2 => 3, tanh),      # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2)
)
# The model encapsulates parameters, randomly initialised. Its initial output is:
out1 = model(noisy)    # 2×1000 Matrix{Float32}, or CuArray{Float32}
probs1 = softmax(out1)    # normalise to get probabilities (and move off GPU)
# To train the model, we use batches of 64 samples, and one-hot encoding:
target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
loader = Flux.DataLoader((noisy, target), batchsize=64, shuffle=true);
opt_state = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.
# Training loop, using the whole data set 1000 times:
losses = []
@inbounds for epoch in 1:1_000
    for xy_cpu in loader
        # Unpack batch of data, and move to GPU:
        x, y = xy_cpu
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat, y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end
opt_state # parameters, momenta and output have all changed
out2 = model(noisy)         # first row is prob. of true, second row p(false)
probs2 = softmax(out2)
mean((probs2[1, :] .> 0.5) .== truth)  # accuracy 94% so far!
