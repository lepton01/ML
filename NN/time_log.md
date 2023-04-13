# Time logs

This file contains basic information about performance of the models and their training.

The simplified notation reads as follows:

* 2-N(2)-n(mr)-1sig

This means that the input vector is 2 rows, followed by a ``normBatch`` layer size of the input, followed by n layers of m neurons each, 'r' denoting the ``relu`` activation function. At the end there is 1 output neuron with ``sigmoid`` activation function.

***

## Model 2-2N-5(256r)-1sig

``31.697556 seconds (49.66 M allocations: 3.385 GiB, 1.54% gc time, 3.39% compilation time) - 70.0%``\
``30.427596 seconds (46.56 M allocations: 3.222 GiB, 1.57% gc time) - 71.6%``\
``30.932998 seconds (46.59 M allocations: 3.223 GiB, 1.50% gc time) - 71.6%``\
``31.204084 seconds (46.55 M allocations: 3.222 GiB, 1.44% gc time) - 73.4%``\
``31.655717 seconds (46.60 M allocations: 3.223 GiB, 1.46% gc time) - 71.8%``

### 1000 n per 'batch'

``52.130530 seconds (47.00 M allocations: 3.229 GiB, 1.07% gc time) - 85.5%``\
``52.198616 seconds (47.04 M allocations: 3.230 GiB, 1.05% gc time) - 84.6%``\
``53.507115 seconds (47.02 M allocations: 3.229 GiB, 1.01% gc time) - 86.5%``\
``51.696379 seconds (47.02 M allocations: 3.229 GiB, 1.06% gc time) - 85.3%``\
``54.409082 seconds (47.02 M allocations: 3.229 GiB, 1.02% gc time) - 85.9%``

***

## Model 2-2N-10(128r)-1sig

``26.742449 seconds (78.58 M allocations: 5.546 GiB, 2.81% gc time, 3.94% compilation time) - 70.39%``\
``25.871774 seconds (76.17 M allocations: 5.413 GiB, 2.86% gc time) - 72.6%``\
``25.875591 seconds (76.12 M allocations: 5.411 GiB, 2.80% gc time) - 69.0%``\
``25.535802 seconds (76.14 M allocations: 5.412 GiB, 2.81% gc time) - 63.4%``\
``25.692058 seconds (76.14 M allocations: 5.413 GiB, 2.85% gc time) - 66.6%``

***

## Model 2-2N-4(512r)-1sig

### 500 n per 'batch'

``56.901125 seconds (44.49 M allocations: 2.999 GiB, 0.81% gc time, 1.46% compilation time) - 71.0%``\
``55.335128 seconds (42.01 M allocations: 2.863 GiB, 0.76% gc time) - 77.2%``\
``58.040763 seconds (42.03 M allocations: 2.864 GiB, 0.76% gc time) - 77.4%``\
``56.710711 seconds (42.00 M allocations: 2.863 GiB, 0.74% gc time) - 77.8%``\
``57.162129 seconds (42.00 M allocations: 2.863 GiB, 0.75% gc time) - 79.2%``

### 1000 ns per 'batch'

``83.926185 seconds (42.39 M allocations: 2.869 GiB, 0.64% gc time) - 89.0%``\
``83.870246 seconds (42.39 M allocations: 2.869 GiB, 0.61% gc time) - 90.3%``\
``83.790833 seconds (42.40 M allocations: 2.869 GiB, 0.61% gc time) - 87.8%``\
``86.590092 seconds (42.40 M allocations: 2.869 GiB, 0.60% gc time) - 89.6%``\
``85.640043 seconds (42.38 M allocations: 2.869 GiB, 0.60% gc time) - 88.4%``

***
***

## Creation and training after `activation` error

### Model 5

Structure:

````jl
model = Chain(
        BatchNorm(2),
        Dense(2 => 512, relu),
        Dense(512 => 512, relu),
        Dense(512 => 512, relu),
        Dense(512 => 512, relu),
        Dense(512 => 512, relu),
        Dense(512 => 512, relu),
        Dense(512 => 512, relu),
        Dense(512 => 512, relu),
        Dense(512 => 1)
    )
````

Trained with 1000 elements of `x` and `a`:\
`178.459372 seconds (77.12 M allocations: 5.138 GiB, 0.69% gc time, 0.85% compilation time) - 81.6%`\
`174.849582 seconds (71.72 M allocations: 4.857 GiB, 0.65% gc time) - 80.9%`\
`170.472582 seconds (71.82 M allocations: 4.860 GiB, 0.68% gc time) - 80.5%`\
`171.117404 seconds (71.82 M allocations: 4.860 GiB, 0.66% gc time) - 78.6%`\
`170.619249 seconds (71.82 M allocations: 4.860 GiB, 0.65% gc time) - 78.4%`

### Model 6

Composed by:

````jl
model = Chain(
        BatchNorm(2),
        Dense(2 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1)
    ) |> gpu
````

Trained with 1000 elements of `x` and `a`:\
`320.676879 seconds (143.15 M allocations: 8.408 GiB, 0.57% gc time, 7.57% compilation time) - 83.2%`\
`302.685313 seconds (50.74 M allocations: 3.691 GiB, 0.32% gc time) - 85.8%`\
`305.424073 seconds (50.74 M allocations: 3.691 GiB, 0.34% gc time) - 85.5%`\
`309.525026 seconds (50.73 M allocations: 3.690 GiB, 0.34% gc time) - 84.2`\
`315.578199 seconds (50.74 M allocations: 3.691 GiB, 0.31% gc time) - 81.3%`

### Model 7

Composed by:

````jl
model = Chain(
        BatchNorm(2),
        Dense(2 => 1024, celu),
        Dense(1024 => 1024, celu),
        Dense(1024 => 1024, celu),
        Dense(1024 => 1024, celu),
        Dense(1024 => 1024, celu),
        Dense(1024 => 1)
    ) |> gpu
````

Trained with 1000 elements of `x` and `a`:\
`269.072777 seconds (58.28 M allocations: 4.098 GiB, 0.43% gc time, 0.61% compilation time) - 82.8%`\
`266.389997 seconds (51.51 M allocations: 3.751 GiB, 0.41% gc time) - 46.5%`\
`287.154505 seconds (51.51 M allocations: 3.751 GiB, 0.38% gc time) - 5.5%`\
`281.704087 seconds (51.51 M allocations: 3.751 GiB, 0.36% gc time) - 4.3%`\
`281.997154 seconds (51.51 M allocations: 3.751 GiB, 0.36% gc time) - 3.0%`

***
***

## After important changes

### NN\model_0

Structure:

````jl
    model = Chain(
        BatchNorm(2),
        Dense(2 => 2056, relu),
        Dense(2056 => 2056, relu),
        Dense(2056 => 2056, relu),
        Dense(2056 => 1)
    )
````

Trained with 1_000 random elements for `a` and `x` for 5_000 epochs.\
`473.212406 seconds (37.23 M allocations: 3.157 GiB, 0.24% gc time, 0.03% compilation time) - 88.7%`\
`285.751118 seconds (112.79 M allocations: 6.757 GiB, 0.60% gc time, 8.81% compilation time) - 90.0%`

### NN\model_1

Structure:

````jl
    model = Chain(
        BatchNorm(2),
        Dense(2 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1024, relu),
        Dense(1024 => 1)
    )
````

Trained with 1000 elements of `x` and `a` for ``1_000 epochs``:\
@time to do: `67.507691 seconds (17.03 M allocations: 1.902 GiB, 1.04% gc time, 3.84% compilation time) - 69.1%`.\
@time to do: `65.906962 seconds (8.63 M allocations: 1.469 GiB, 0.60% gc time) - 86.0%`.\
@time to do: `84.268995 seconds (8.63 M allocations: 1.469 GiB, 0.52% gc time) - 86.1%`.\
@time to do: `67.337913 seconds (8.63 M allocations: 1.469 GiB, 0.36% gc time) - 85.9%`.\
@time to do: `79.021504 seconds (8.63 M allocations: 1.469 GiB, 0.51% gc time) - 85.9%`.

Trained with 5000 elements of ``x`` and ``a``, ``10_000`` epochs:
``@time: 511.231058 seconds (229.61 M allocations: 13.867 GiB, 0.66% gc time, 7.18% compilation time: 0% of which was recompilation)``.\
``@time: 1512.553041 seconds (87.91 M allocations: 6.607 GiB, 0.42% gc time, 0.01% compilation time) - 94.48%``.

### NN\model_2

Structure:

````jl
    model = Chain(
        BatchNorm(2),
        Dense(2 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1024),
        Dense(1024 => 1)
    )
````

Trained:
@time: `676.933945 seconds (86.39 M allocations: 6.532 GiB, 0.32% gc time) - 84.3%`.
@time: `849.352302 seconds (86.40 M allocations: 6.533 GiB, 0.21% gc time) - 85.39%`.
