

"""
BasicParticleFilter2(predict_model, reweight_model, resampler, n_init::Integer, rng::AbstractRNG)
    BasicParticleFilter2(model, resampler, n_init::Integer, rng::AbstractRNG)

Construct a basic particle filter with three steps: predict, reweight, and resample.

In the second constructor, `model` is used for both the prediction and reweighting.
"""
mutable struct BasicParticleFilter2{PM,RM,RS,RNG<:AbstractRNG,PMEM} <: ParticleFilters.Updater
    predict_model::PM
    reweight_model::RM
    resampler::RS
    n_init::Int
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

## Constructors ##
function BasicParticleFilter2(model, resampler, n::Integer, rng::AbstractRNG=Random.GLOBAL_RNG)
    return BasicParticleFilter2(model, model, resampler, n, rng)
end

function BasicParticleFilter2(pmodel, rmodel, resampler, n::Integer, rng::AbstractRNG=Random.GLOBAL_RNG)
    return BasicParticleFilter2(pmodel,
                               rmodel,
                               resampler,
                               n,
                               rng,
                               particle_memory(pmodel),
                               Float64[]
                              )
end


function ParticleFilters.update(up::BasicParticleFilter2, b::ParticleCollection, ctx, o)
    pm = up._particle_memory
    wm = up._weight_memory

    resize!(pm, n_particles(b))
    resize!(wm, n_particles(b))
    predict!(pm, up.predict_model, b, ctx, o, up.rng)
    reweight!(wm, up.reweight_model, b, ctx, pm, o, up.rng)

    weight_std = wm ./ sum(wm)

    num_eff = sum(weight_std) / sum(weight_std .^ 2)

    println("N_eff = $num_eff")

    resampled_particles = resample(
        up.resampler,
        WeightedParticleBelief(pm, wm, sum(wm), nothing),
        up.predict_model,
        up.reweight_model,
        b, ctx, o,
        up.rng
    )

    weights_after_resample = get_weights(up.reweight_model, resampled_particles, ctx, o)

    post_regularised_particles = post_regularise(
        resampled_particles, weights_after_resample, up.rng
    )

    return post_regularised_particles
end

function Random.seed!(f::BasicParticleFilter2, seed)
    Random.seed!(f.rng, seed)
    return f
end

ParticleFilters.predict(f::BasicParticleFilter2, args...) = ParticleFilters.predict(f.predict_model, args...)
