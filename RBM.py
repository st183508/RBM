import numpy as np
import matplotlib.pyplot as plt

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
        a_plus = a.copy()
        a_minus = a.copy()
        a_plus[i] += step_size
        a_minus[i] -= step_size
        log_derivative_a[i] = (np.log(rbm_ansatz(s, a_plus, b, W)) - np.log(rbm_ansatz(s, a_minus, b, W))) / (2 * step_size)


    for i in range(M):
        b_plus = b.copy()
        b_minus = b.copy()
        b_plus[i] += step_size
        b_minus[i] -= step_size
        log_derivative_b[i] = (np.log(rbm_ansatz(s, a, b_plus, W)) - np.log(rbm_ansatz(s, a, b_minus, W))) / (2 * step_size)


    for i in range(M):
        for j in range(N):
            W_plus = W.copy()
            W_minus = W.copy()
            W_plus[i, j] += step_size
            W_minus[i, j] -= step_size
            log_derivative_W[i, j] = (np.log(rbm_ansatz(s, a, b, W_plus)) - np.log(rbm_ansatz(s, a, b, W_minus))) / (2 * step_size)

    return log_derivative_a * rbm_ansatz(s,a,b,W), log_derivative_b * rbm_ansatz(s,a,b,W), log_derivative_W *  rbm_ansatz(s,a,b,W)


def random_state_amp(N):
    dim = 2 ** N
    state_amp = np.random.rand(dim) + 1j * np.random.rand(dim)

    return state_amp 















def exaxct_dericative(rbm_ansatz, s ,a ,b ,W ):
    log_derivative_a = np.zeros(N)
    log_derivative_b = np.zeros(M)
    log_derivative_W = np.zeros((M,N))

    for i in range(N): 
        log_derivative_a[i] = s[i]

    for i in range(M):
        log_derivative_b[i] = np.tanh(b[i] + np.dot(W[i], s))

    for i in range(M):
        for j in range(N):
            log_derivative_W[i,j] = s[j] * np.tanh(b[i] + np.dot(W[i], s))

    return log_derivative_a * rbm_ansatz(s,a,b,W), log_derivative_b * rbm_ansatz(s,a,b,W), log_derivative_W *  rbm_ansatz(s,a,b,W)


def plot_derivative_comparison(s, a, b, W):
    numerical = derivative(rbm_ansatz, s, a, b, W)
    analytical = exaxct_dericative(rbm_ansatz, s, a, b, W)
    parameter_names = (
        [f"a[{i}]" for i in range(N)],
        [f"b[{i}]" for i in range(M)],
        [f"W[{i},{j}]" for i in range(M) for j in range(N)],
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    titles = ("Ableitung nach a", "Ableitung nach b", "Ableitung nach W")

    for axis, numeric_values, analytical_values, labels, title in zip(
        axes, numerical, analytical, parameter_names, titles
    ):
        numeric_values = np.ravel(numeric_values)
        analytical_values = np.ravel(analytical_values)
        positions = np.arange(len(labels))
        width = 0.38

        axis.bar(positions - width / 2, numeric_values, width, label="Numerisch")
        axis.bar(positions + width / 2, analytical_values, width, label="Analytisch")
        axis.set_title(f"{title}\nmax. Abweichung: {np.max(np.abs(numeric_values - analytical_values)):.2e}")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Ableitungswert")
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend()
    fig.suptitle("Numerische und analytische RBM-Ableitungen")
    return fig, axes


if __name__ == "__main__":
    s = np.array([1.0, -1.0])
    a = np.array([0.2, -0.3])
    b = np.array([0.1, 0.4])
    W = np.array([[0.3, -0.2], [0.5, 0.1]])
    plot_derivative_comparison(s, a, b, W)
    plt.show()





 
