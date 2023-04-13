# Time logs 2

File containing information and logs related to the training of the models after the discovery of a bug in the training data that prevented success.

***

## model_0

### Structure

````jl
    model = Chain(
        BatchNorm(2),
        Dense(2 => 2056, relu),
        Dense(2056 => 2056, relu),
        Dense(2056 => 2056, relu),
        Dense(2056 => 1)
    )
````

a

***

## model_1

a

***

## model_2

a

***

## model_3
