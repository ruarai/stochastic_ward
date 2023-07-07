
global gp_mat_size::Int64 = 64

struct gp_state
    distribution::Gaussian

    x::Vector{Float64}
end

# Matern 5/2 covariance function
function covar_matern(x_a, x_b, variance, l)
    dx = abs(x_a - x_b)

    return variance * (
        1 + 
        (sqrt(5) * dx) / l +
        (5 * dx ^ 2) / (3 * l ^ 2)
    ) *
    exp(-(sqrt(5) * dx) / l)
end

function make_gp(variance, lengthscale)
    Σ = zeros(gp_mat_size, gp_mat_size)

    for i in 1:gp_mat_size, j in 1:gp_mat_size
        Σ[i, j] = covar_matern(i, j, variance, lengthscale)

    end

    dist = Gaussian(zeros(gp_mat_size), Σ)

    x0 = rand(dist)

    return gp_state(dist, x0)
end


function step_gp(gp)
    X_conditional = GaussianDistributions.conditional(
        gp.distribution,
        [gp_mat_size],
        1:(gp_mat_size - 1),
        gp.x[2:gp_mat_size]
    )

    x = vcat(gp.x[2:gp_mat_size], rand(X_conditional))
        
    return gp_state(
        gp.distribution,
        x
    )
end
