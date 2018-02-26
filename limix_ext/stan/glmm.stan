data {
    int<lower=0> N; // number of samples
    int<lower=0> P; // number of covariates
    int nsuc[N];
    int ntri[N];
    matrix[N, P] X; // covariates
    matrix[N, N] K; // kinship
}
transformed data {
    matrix[N, N] L;
    L = cholesky_decompose(K);
}
parameters {
    vector[P] effsiz;
    vector[N] u_effsiz;
    vector[N] e;
    real<lower=0.0001> sigma_g;
    real<lower=0.0001> sigma_e;
}
transformed parameters {
    vector[N] u;
    u  = L * u_effsiz;
}
model {
    vector[N] z;
    u_effsiz ~ normal(0, sigma_g);
    e ~ normal(0, sigma_e);
    z = X * effsiz + u + e;
    nsuc ~ binomial_logit(ntri, z);
}
