function get_weights(m::ReweightModel, particles, ctx, y)
    weights = Vector{Float64}(undef, n_particles(particles))

    for i in 1:n_particles(particles)
        x1 = particle(particles, i)
        # First parameter is not used!
        weights[i] = m.g(x1, ctx, x1, y)
    end

    return weights
end