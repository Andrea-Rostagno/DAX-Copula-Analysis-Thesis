import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1️⃣ Caricamento e preprocessing dei dati
df = pd.read_csv('DAX_3Y-1M.csv', index_col='DateTime', parse_dates=True, low_memory=False)

# Conversione dei dati numerici e pulizia
cols = ['Open', 'Close']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Calcolo dei rendimenti logaritmici
returns = np.log(df / df.shift(1)).dropna()
u, v = returns['Open'], returns['Close']

# Calcolo della correlazione standard per confronto
pearson_corr = np.corrcoef(u, v)[0, 1]
print(f"📊 Correlazione di Pearson tra Open e Close: {pearson_corr:.4f}")

# Trasformazione in distribuzione uniforme
u_unif = stats.rankdata(u) / (len(u) + 1)
v_unif = stats.rankdata(v) / (len(v) + 1)


# 2️⃣ Definizione della funzione di verosimiglianza per la Copula Gaussiana
def gaussian_copula_log_likelihood(rho, u, v):
    """Calcola la log-verosimiglianza della copula gaussiana."""
    if abs(rho) >= 0.999:  # Evita problemi numerici ai bordi
        return -np.inf

    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)

    # Log-verosimiglianza della copula gaussiana
    term1 = -0.5 * np.log(1 - rho ** 2)
    term2 = -(rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * (1 - rho ** 2))

    return np.sum(term1 + term2)


# Funzione negativa per minimizzazione
def neg_log_likelihood(rho, u, v):
    """Versione negativa della log-verosimiglianza per minimizzazione."""
    return -gaussian_copula_log_likelihood(rho[0], u, v)


# 3️⃣ Stima MLE per confronto e inizializzazione
print("\n====== STIMA MASSIMA VEROSIMIGLIANZA ======")
initial_guess = [pearson_corr]  # Usiamo la correlazione di Pearson come punto di partenza
bounds = [(-0.999, 0.999)]  # Limiti per il parametro rho

result = minimize(neg_log_likelihood, initial_guess, args=(u_unif, v_unif),
                  bounds=bounds, method='L-BFGS-B')

mle_rho = result.x[0]
print(f"📊 Stima MLE di ρ: {mle_rho:.4f}")
print(f"📊 Log-verosimiglianza massimizzata: {-result.fun:.4f}")
print(f"📊 Stato dell'ottimizzazione: {result.success}")


# 4️⃣ Algoritmo MCMC migliorato con multiple chain e diagnostica robusta
def improved_metropolis_hastings(log_likelihood, u, v, n_chains=4, n_iter=20000, burn_in=5000,
                                 target_acceptance=0.234):
    """
    Implementazione migliorata di Metropolis-Hastings con catene multiple
    e adattamento del passo basato su Gelman et al. (1996).
    """
    n_params = 1  # Dimensione del parametro (solo rho)

    # Inizializzazione di catene multiple con punti di partenza diversi
    chains = np.zeros((n_chains, n_iter, n_params))

    # Punti di partenza distribuiti uniformemente tra -0.9 e 0.9
    starting_points = np.linspace(-0.8, 0.8, n_chains)

    # Parametri per l'adattamento
    gamma1 = 0.75  # Parametro che controlla l'adattamento iniziale
    adaptation_steps = int(n_iter * 0.6)  # Numero di passi di adattamento

    # Memorizza i tassi di accettazione e le larghezze delle proposte
    acceptance_rates = np.zeros(n_chains)
    proposal_widths = np.ones(n_chains) * 0.1  # Valori iniziali

    print("\n====== AVVIO MCMC CON MULTIPLE CATENE ======")

    for c in range(n_chains):
        chain = chains[c]
        chain[0, 0] = starting_points[c]  # Punto di partenza
        accepted = 0

        print(f"- Catena {c + 1}: punto di partenza = {starting_points[c]:.4f}")

        for i in range(1, n_iter):
            current_rho = chain[i - 1, 0]

            # Passo adattivo che diminuisce con le iterazioni
            if i <= adaptation_steps:
                adapt_factor = (i / adaptation_steps) ** gamma1
                current_width = proposal_widths[c] * (1 - adapt_factor) + 0.1 * adapt_factor
            else:
                current_width = proposal_widths[c]

            # Proposta: U(-δ,δ) centrata sull'attuale valore
            delta = current_width
            proposal = current_rho + np.random.uniform(-delta, delta)

            # Rifletti se fuori dai limiti (-0.999, 0.999)
            if proposal <= -0.999:
                proposal = -0.999 + ((-0.999) - proposal)
            elif proposal >= 0.999:
                proposal = 0.999 - (proposal - 0.999)

            # Calcola il rapporto di accettazione
            log_p_current = log_likelihood(current_rho, u, v)
            log_p_proposal = log_likelihood(proposal, u, v)
            log_accept_ratio = log_p_proposal - log_p_current

            # Accetta o rifiuta
            if np.log(np.random.random()) < log_accept_ratio:
                chain[i, 0] = proposal
                accepted += 1
            else:
                chain[i, 0] = current_rho

            # Adatta la larghezza della proposta
            if i % 500 == 0 and i <= adaptation_steps:
                batch_acceptance = accepted / i

                # Aggiusta la larghezza per avvicinarsi al tasso di accettazione target
                if batch_acceptance > target_acceptance:
                    proposal_widths[c] *= 1.1  # Aumenta
                else:
                    proposal_widths[c] *= 0.9  # Diminuisci

                # Limita la larghezza per sicurezza
                proposal_widths[c] = max(0.01, min(1.0, proposal_widths[c]))

                if i % 5000 == 0:
                    print(f"  Iterazione {i}, catena {c + 1}: accettazione = {batch_acceptance:.4f}, "
                          f"larghezza = {proposal_widths[c]:.4f}")

        # Statistiche finali per questa catena
        acceptance_rates[c] = accepted / n_iter
        print(f"- Catena {c + 1} completata: accettazione = {acceptance_rates[c]:.4f}, "
              f"larghezza finale = {proposal_widths[c]:.4f}")

    # Rimuovi burn-in e unisci le catene per l'analisi
    combined_samples = chains[:, burn_in:, 0].flatten()

    return chains, acceptance_rates, combined_samples


# 5️⃣ Esegui MCMC migliorato
n_chains = 4
n_iter = 20000
burn_in = 5000

chains, acceptance_rates, combined_samples = improved_metropolis_hastings(
    gaussian_copula_log_likelihood,
    u_unif,
    v_unif,
    n_chains=n_chains,
    n_iter=n_iter,
    burn_in=burn_in
)


# 6️⃣ Diagnostica di Gelman-Rubin
def gelman_rubin(chains, burn_in=0):
    """Calcola la diagnostica R-hat di Gelman-Rubin."""
    n_chains, n_iter, n_params = chains.shape

    if burn_in > 0:
        chains = chains[:, burn_in:, :]

    n_iter = chains.shape[1]

    # Medie delle catene
    chain_means = np.mean(chains, axis=1)  # shape: (n_chains, n_params)

    # Varianza tra le catene
    B = n_iter * np.var(chain_means, axis=0, ddof=1)  # shape: (n_params,)

    # Varianza entro le catene
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)  # shape: (n_params,)

    # Stima complessiva della varianza
    var_hat = ((n_iter - 1) / n_iter) * W + (1 / n_iter) * B

    # Calcolo R-hat
    R_hat = np.sqrt(var_hat / W)

    return R_hat


r_hat = gelman_rubin(chains, burn_in=0)
print(f"\n📊 Diagnostica di Gelman-Rubin (R-hat): {r_hat[0]:.4f}")

# 7️⃣ Statistiche sulle catene unite (post burn-in)
rho_mean = np.mean(combined_samples)
rho_std = np.std(combined_samples)
rho_median = np.median(combined_samples)
rho_ci = np.percentile(combined_samples, [2.5, 97.5])

print("\n====== RISULTATI FINALI ======")
print(f"📊 Stima Bayesiana di ρ con MCMC: {rho_mean:.4f} ± {rho_std:.4f}")
print(f"📊 Mediana: {rho_median:.4f}")
print(f"📊 Intervallo di credibilità al 95%: [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}]")
print(f"📊 Dimensione del campione efficace: {len(combined_samples)}")
print(f"📊 Confronto con MLE: {mle_rho:.4f}")
print(f"📊 Confronto con correlazione di Pearson: {pearson_corr:.4f}")

# 8️⃣ Visualizzazioni migliorate
plt.figure(figsize=(15, 12))

# Plot 1: Trace plot di tutte le catene
plt.subplot(2, 2, 1)
for c in range(n_chains):
    plt.plot(chains[c, burn_in:, 0], alpha=0.7, label=f"Catena {c + 1}")
plt.axhline(rho_mean, color='red', linestyle='dashed', label=f"Media: {rho_mean:.4f}")
plt.axhline(mle_rho, color='green', linestyle='dotted', label=f"MLE: {mle_rho:.4f}")
plt.xlabel("Iterazione (dopo burn-in)")
plt.ylabel("Valore di ρ")
plt.title("Trace Plot delle catene MCMC")
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Istogramma con kernel density
plt.subplot(2, 2, 2)
plt.hist(combined_samples, bins=50, density=True, alpha=0.6, color='blue')
# Kernel density estimate
from scipy.stats import gaussian_kde

kde = gaussian_kde(combined_samples)
x = np.linspace(-0.99, 0.99, 1000)
plt.plot(x, kde(x), 'r-', lw=2, label='KDE')
plt.axvline(rho_mean, color='red', linestyle='dashed', label=f"Media: {rho_mean:.4f}")
plt.axvline(mle_rho, color='green', linestyle='dotted', label=f"MLE: {mle_rho:.4f}")
plt.axvline(rho_ci[0], color='orange', linestyle='dotted')
plt.axvline(rho_ci[1], color='orange', linestyle='dotted')
plt.xlabel("Valore di ρ")
plt.ylabel("Densità")
plt.title(f"Distribuzione Posteriore di ρ (95% CI: [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}])")
plt.legend()
plt.grid(alpha=0.3)

# Plot 3: Grafico della funzione di verosimiglianza
plt.subplot(2, 2, 3)
rho_range = np.linspace(-0.95, 0.95, 200)
log_likes = [gaussian_copula_log_likelihood(r, u_unif, v_unif) for r in rho_range]
plt.plot(rho_range, log_likes, 'b-', lw=2)
plt.axvline(mle_rho, color='green', linestyle='dotted', label=f"MLE: {mle_rho:.4f}")
plt.axvline(rho_mean, color='red', linestyle='dashed', label=f"Media post.: {rho_mean:.4f}")
plt.xlabel("Valore di ρ")
plt.ylabel("Log-verosimiglianza")
plt.title("Funzione di Log-verosimiglianza")
plt.legend()
plt.grid(alpha=0.3)

# Plot 4: Running means per catena
plt.subplot(2, 2, 4)
for c in range(n_chains):
    chain_samples = chains[c, burn_in:, 0]
    running_mean = np.cumsum(chain_samples) / np.arange(1, len(chain_samples) + 1)
    plt.plot(running_mean, alpha=0.7, label=f"Catena {c + 1}")
plt.axhline(rho_mean, color='red', linestyle='dashed', label=f"Media finale: {rho_mean:.4f}")
plt.xlabel("Iterazione (dopo burn-in)")
plt.ylabel("Media cumulativa")
plt.title("Convergenza delle medie")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mcmc_diagnostics.png', dpi=300)
plt.show()

# 9️⃣ Plot della Copula stimata
plt.figure(figsize=(10, 8))

# Generiamo punti dalla copula gaussiana stimata
n_points = 2000
rho_est = rho_mean

# Generiamo numeri casuali da una distribuzione normale bivariata con correlazione rho_est
mean = [0, 0]
cov = [[1, rho_est], [rho_est, 1]]
random_normal = np.random.multivariate_normal(mean, cov, n_points)

# Trasformiamo in variabili uniformi tramite la CDF
random_copula = stats.norm.cdf(random_normal)

# Scatter plot dei dati originali
plt.subplot(1, 2, 1)
plt.scatter(u_unif, v_unif, alpha=0.5, s=8, c='blue', label='Dati originali')
plt.title(f"Dati originali (Pearson corr = {pearson_corr:.4f})")
plt.xlabel("Rango uniforme - Open")
plt.ylabel("Rango uniforme - Close")
plt.grid(alpha=0.3)
plt.legend()

# Scatter plot della copula stimata
plt.subplot(1, 2, 2)
plt.scatter(random_copula[:, 0], random_copula[:, 1], alpha=0.5, s=8, c='red',
            label=f'Copula Gaussiana (ρ={rho_est:.4f})')
plt.title(f"Copula Gaussiana stimata (ρ={rho_est:.4f})")
plt.xlabel("U1")
plt.ylabel("U2")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('copula_comparison.png', dpi=300)
plt.show()

