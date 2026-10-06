import numpy as np

N = 2
M = 2

def rbm_ansatz(s,a,b,W):

    exponetial_term = np.exp(np.dot(a,s))

    cosh_term = 1

    for i in range(M):
        cosh_term *= np.cosh(b[i] + np.dot(W[i], s))

    return exponetial_term * cosh_term


def  derivative(rbm_ansatz, s ,a ,b ,W ):

    step_size = 1e-5

    log_derivative_a = np.zeros(N)
    log_derivative_b = np.zeros(M)
    log_derivative_W = np.zeros((M,N))

    for i in range(N): 
        log_derivative_a[i] =  (np.log(rbm_ansatz(s,a + step_size,b,W)) -  np.log(rbm_ansatz(s,a - step_size,b,W)))/(2*step_size)


    for i in range(M):
        log_derivative_b[i] =  (np.log(rbm_ansatz(s,a,b + step_size,W)) -  np.log(rbm_ansatz(s,a,b - step_size,W)))/(2*step_size)


    for i in range(M):
        for j in range(N):
            log_derivative_W[i,j] =  (np.log(rbm_ansatz(s,a,b,W + step_size*np.eye(M,N)[i])) -  np.log(rbm_ansatz(s,a,b,W - step_size*np.eye(M,N)[i])))/(2*step_size)

    return log_derivative_a * rbm_ansatz(s,a,b,W), log_derivative_b * rbm_ansatz(s,a,b,W), log_derivative_W *  rbm_ansatz(s,a,b,W)


 
