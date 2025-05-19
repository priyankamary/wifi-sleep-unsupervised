import pymc3 as pm

def run_model(data, T1_prior_fn, T2_prior_fn, model_name):
    with pm.Model(name=model_name) as model:
        T1 = T1_prior_fn()
        T2 = T2_prior_fn()
        rate1 = pm.Gamma("rate1", alpha=2.5, beta=1.0)
        rate2 = pm.Exponential("rate2", lam=0.0001)

        rate_sleep = pm.math.switch(T1 >= data['bin'].values, rate1, rate2)
        rate_awake = pm.math.switch(T2 >= data['bin'].values, rate2, rate1)

        pm.Poisson("sleep", rate_sleep, observed=data['connected'].values)
        pm.Poisson("awake", rate_awake, observed=data['connected'].values)

        trace = pm.sample(1000, tune=1000, cores=1, progressbar=False)
    return trace, model
