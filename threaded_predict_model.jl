
struct ThreadedPredictModel{S, F}
    f::F
end

ThreadedPredictModel{S}(f::F) where {S, F<:Function} = ThreadedPredictModel{S, F}(f)

function ParticleFilters.predict!(pm, m::ThreadedPredictModel, b, u, rng)
    Threads.@threads for i in 1:ParticleFilters.n_particles(b)
        rng_thread = Random.MersenneTwister()
        x1 = particle(b, i)
        pm[i] = m.f(x1, u, rng_thread)
    end
end

ParticleFilters.particle_memory(m::ThreadedPredictModel{S}) where S = S[]

