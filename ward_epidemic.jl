

struct ward_epidemic
    S::Int64
    I::Int64
    Q::Int64
end


function step_ward_epidemic(x, cases)
    n_steps = 1000
    dt = 1 / n_steps

    for t in 1:n_steps
        rate_new_S = 1
        rate_new_I = cases * 1e-4
    
        rate_S_to_I = x.I * 0.9 / (x.I + x.S + 1e-10)
        rate_I_to_Q = 0.5
        rate_death_Q = 1 / 7
    
    
        new_S = rand(Poisson(rate_new_S * dt))
        S_to_I = rand(Binomial(x.S, expm1(rate_S_to_I * dt)))
    
        new_I = rand(Poisson(rate_new_I * dt))
    
        I_to_Q = rand(Binomial(x.I, expm1(rate_I_to_Q * dt)))
        death_Q = rand(Binomial(x.Q, rate_death_Q * dt))
    
        x = ward_epidemic(
            x.S + new_S - S_to_I,
            x.I + new_I + S_to_I - I_to_Q,
            x.Q + I_to_Q - death_Q
        )
    end

    return x
end
