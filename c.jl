

using Distributions
using StatsFuns
using DataFrames
using Random

using ParticleFilters

struct ThreadedPredictModel{S, F}
    f::F
end

ThreadedPredictModel{S}(f::F) where {S, F<:Function} = ThreadedPredictModel{S, F}(f)

function ParticleFilters.predict!(pm, m::ThreadedPredictModel, b, u, rng)
    Threads.@threads for i in 1:ParticleFilters.n_particles(b)
        x1 = particle(b, i)
        pm[i] = m.f(x1, u, rng)
    end
end

ParticleFilters.particle_memory(m::ThreadedPredictModel{S}) where S = S[]



dynamics(x, u, rng) = x + u + randn(rng)
y_likelihood(x_previous, u, x, y) = pdf(Normal(), y - x)

model = ParticleFilterModel{Float64}(dynamics, y_likelihood)


num_particles = 100


predict_model = ThreadedPredictModel{Float64}(dynamics)

reweight_model = ReweightModel(y_likelihood)

resample_model = LowVarianceResampler(num_particles)

particle_filter = BasicParticleFilter(predict_model, reweight_model, resample_model, num_particles, MersenneTwister())


b = ParticleCollection([1.0, 2.0, 3.0, 4.0])
u = 1.0
y = 3.0

b_new = update(particle_filter, b, u, y)