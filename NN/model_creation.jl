#01/04/2023
"""
    bessel_model_creation(model_name)

Creation of a NN model to save on file: `model_name.bson`.

Do not add `.bson` to the string input, as the function already does.
"""
function bessel_model_creation(model_name::String)
    model::Chain = Chain(
        BatchNorm(2),
        Dense(2 => 512, relu),
        Dense(512 => 512),
        Dense(512 => 1)
    )
    BSON.@save model_name*".bson" model
    return 
end
