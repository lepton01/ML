#16/03/2023
using Statistics, GLMakie
function generate_real_data(n)
    x1 = rand(1, n) .- 0.5
    x2 = (x1 .* x1) * 3 .+ randn(1, n) * 0.1
    vcat(x1, x2)
end
function generate_fake_data(n)
    θ = 2π * rand(1, n)
    r = rand(1, n) / 3
    x1 = @. r * cos(θ)
    x2 = @. r * sin(θ) + 0.5
    vcat(x1, x2)
end
train_size::Int = 20000
real = generate_real_data(train_size)
fake = generate_fake_data(train_size)
#=
scatter(real[1, 1:500], real[2, 1:500],
    xlabel = "x",
    ylabel = "y",
    legend = false
)
scatter!(fake[1, 1:500], fake[2, 1:500])
=#
f1 = Figure()
ax1 = Axis(f1[1, 1], title="a")
scatter!(ax1, real[1, 1:500], real[2, 1:500], color=:green,label="Real")
scatter!(ax1, fake[1, 1:500], fake[2, 1:500], color=:red, label="Fake")
axislegend()
f1
