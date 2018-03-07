library(glmm)

rglmm_binomial <- function(nsuc, ntri, X, K, npoints, G) {

    if (missing(npoints)) {
        npoints <- 10**4
    }

    y <- format_outcome(nsuc, ntri)
    X <- format_covariates(X)
    K <- format_covariance(K)
    E <- format_noise(nrow(K))

    data <- cbind(y, X, K, E)

    of <- outcome_formula(X)
    rf <- random_formula(K)
    nf <- ~ 0 + N
    doPQL <- TRUE

    model = glmm(of, random=list(rf, nf),
                 varcomps.names=list("v_g", "v_e"), data=data,
                 family.glmm=binomial.glmm,
                 m=npoints, doPQL=doPQL)

    intercept <- coef(model)["(Intercept)"]
    effsizes <- coef(model)[colnames(X)]
    v_g <- model$nu["v_g"]
    v_e <- model$nu["v_e"]
    lml = model$likelihood.value

    r = list("v_g"=v_g, "v_e"=v_e, "intercept"=intercept, effsizes=effsizes,
             lml=lml)

    if (!missing(G)) {
        alt_lmls <- rep(0, ncol(G))
    
        for (i in 1:ncol(G)) {
            candidate = as.data.frame(G[, i])
            colnames(candidate) <- c("candidate")

            Xaug <- cbind(X, candidate)
            of <- outcome_formula(Xaug)
            data <- cbind(y, Xaug, K, E)
            model = glmm(of, random=list(rf, nf),
                         varcomps.names=list("v_g", "v_e"), data=data,
                         family.glmm=binomial.glmm,
                         m=npoints, doPQL=doPQL)

            alt_lmls[i] = model$likelihood.value
        }
        r[["alt_lmls"]] = alt_lmls
    }

    return(r)
}

format_covariates <- function(X) {
    X_design = as.data.frame(X)
    colnames(X_design) <- sprintf("C%s", seq(1:ncol(X)))
    return(X_design)
}

format_covariance <- function(K) {
    QV <- eigen(K, TRUE)
    ok <- QV$values > 1e-5
    vals <- QV$values[ok]
    Q <- QV$vectors[, ok]
    random_design <- Q %*% sqrt(diag(vals))
    colnames(random_design) <- sprintf("R%s", seq(1:ncol(random_design)))
    return(as.data.frame(random_design))
}

outcome_formula <- function(X) {
    p1 <- "cbind(NSUC, NFAI) ~ 1 + "
    p2 <- paste(colnames(X), collapse=" + ")
    return(as.formula(paste(p1, p2)))
}

random_formula <- function(K) {
    p1 <- "~ 0 + "
    p2 <- paste(colnames(K), collapse=" + ")
    return(as.formula(paste(p1, p2)))
}

format_outcome <- function(nsuc, ntri) {
    nfai = ntri - nsuc
    outcome <- cbind(nsuc, nfai)
    colnames(outcome) <- c("NSUC", "NFAI")
    return(outcome)
}

format_noise <- function(n) {
    noise_design <- matrix(1L, nrow=n)
    colnames(noise_design) <- c("N")
    return(as.data.frame(noise_design))
}
