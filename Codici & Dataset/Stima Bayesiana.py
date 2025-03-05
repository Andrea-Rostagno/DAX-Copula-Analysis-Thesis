import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("📊 STIMA BAYESIANA DEI PARAMETRI DELLE COPULE")
print("=" * 70)

# 1. Caricamento e preprocessing dei dati
print("\n1️⃣ Caricamento e preprocessing dei dati")
print("-" * 50)

# Prova a caricare il file pulito, altrimenti usa quello originale con opzioni robuste
try:
    df = pd.read_csv('DAX_cleaned.csv', index_col=0, parse_dates=True)
    print("✅ Caricato file pulito 'DAX_cleaned.csv'")
except FileNotFoundError:
    try:
        print("⚠️ File pulito non trovato, provo con il file originale...")
        # Tentativo con opzioni più robuste
        df = pd.read_csv('DAX_3Y-1M.csv',
                         sep=None,  # Auto-detect del separatore
                         engine='python',
                         encoding='utf-8',
                         on_bad_lines='skip',
                         na_values=['na', 'NA', 'N/A', ''])

        # Se c'è una colonna che sembra una data, usala come indice
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            print(f"✅ Usando '{date_cols[0]}' come colonna datetime")
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df.set_index(date_cols[0], inplace=True)

        print("✅ Caricamento completato con opzioni robuste")
    except Exception as e:
        print(f"❌ Errore nel caricamento del file originale: {e}")
        print("⚠️ Utilizzo di un dataset di esempio per dimostrare il codice...")

        # Creiamo dati di esempio correlati
        np.random.seed(42)
        n = 1000
        rho = 0.7

        # Genera dati con correlazione
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        z = np.random.multivariate_normal(mean, cov, n)

        # Trasforma in prezzi
        price_open = 13000 + 500 * np.cumsum(z[:, 0] * 0.01)
        price_close = 13000 + 500 * np.cumsum(z[:, 1] * 0.01)

        # Crea DataFrame
        dates = pd.date_range(start='2020-01-01', periods=n, freq='15T')
        df = pd.DataFrame({
            'Open': price_open,
            'Close': price_close
        }, index=dates)
        print("✅ Dataset di esempio creato")

# Verifica quali colonne possiamo usare
required_cols = ['Open', 'Close']
available_cols = [col for col in required_cols if col in df.columns]

if len(available_cols) < 2:
    print(f"\n❌ Insufficienti colonne di prezzo. Necessarie almeno Open e Close.")
    # Prova a identificare colonne numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        print(f"✅ Utilizzando le colonne numeriche: {numeric_cols[:2]}")
        df['Open'] = df[numeric_cols[0]]
        df['Close'] = df[numeric_cols[1]]
        available_cols = ['Open', 'Close']

# 2. Calcolo dei rendimenti logaritmici
print("\n2️⃣ Calcolo dei rendimenti logaritmici")
print("-" * 50)

# Assicuriamoci che i dati siano numerici
for col in available_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ordina per indice (importante per dati temporali)
df = df.sort_index()

# Calcola i rendimenti logaritmici
log_returns = np.log(df[available_cols] / df[available_cols].shift(1)).dropna()
print(f"✅ Calcolati {len(log_returns)} rendimenti logaritmici")

# 3. Trasformazione in distribuzioni uniformi
print("\n3️⃣ Trasformazione in distribuzioni uniformi")
print("-" * 50)

u = stats.rankdata(log_returns['Open']) / (len(log_returns) + 1)
v = stats.rankdata(log_returns['Close']) / (len(log_returns) + 1)

print(f"✅ Dati trasformati in distribuzioni uniformi [0,1]")

# 4. Calcolo delle misure di dipendenza
print("\n4️⃣ Misure di dipendenza")
print("-" * 50)

# Correlazione di Pearson
pearson_corr = np.corrcoef(log_returns['Open'], log_returns['Close'])[0, 1]
print(f"Correlazione di Pearson: {pearson_corr:.4f}")

# Tau di Kendall
kendall_tau, p_value = stats.kendalltau(log_returns['Open'], log_returns['Close'])
print(f"Tau di Kendall: {kendall_tau:.4f} (p-value: {p_value:.6f})")

# Rho di Spearman
spearman_rho, p_value = stats.spearmanr(log_returns['Open'], log_returns['Close'])
print(f"Rho di Spearman: {spearman_rho:.4f} (p-value: {p_value:.6f})")

# 5. Implementazione MCMC manuale per stima bayesiana
print("\n5️⃣ Stima bayesiana con MCMC (Metropolis-Hastings)")
print("-" * 50)


# Funzioni di log-densità per la copula Gaussiana
def gaussian_copula_loglik(rho, u, v):
    """Log-verosimiglianza per la copula Gaussiana."""
    if abs(rho) >= 0.999:  # Evitare problemi numerici ai bordi
        return -np.inf

    # Trasformiamo in variabili normali
    norm_u = stats.norm.ppf(u)
    norm_v = stats.norm.ppf(v)

    # Log-verosimiglianza per la copula Gaussiana
    term1 = -0.5 * np.log(1 - rho ** 2)
    term2 = -0.5 * (rho ** 2 * (norm_u ** 2 + norm_v ** 2) - 2 * rho * norm_u * norm_v) / (1 - rho ** 2)

    return np.sum(term1 + term2)


# Prior log-densità per rho (distribuzione Beta trasformata)
def log_prior_rho(rho, alpha, beta):
    """Log-densità prior per rho (distribuzione Beta trasformata in [-1, 1])."""
    if abs(rho) >= 1:
        return -np.inf

    # Trasforma rho da [-1, 1] a [0, 1] per la distribuzione Beta
    rho_01 = (rho + 1) / 2

    # Log-densità della Beta
    log_density = stats.beta.logpdf(rho_01, alpha, beta)

    # Aggiusta per la trasformazione (Jacobiano)
    return log_density + np.log(0.5)  # +log(0.5) per la trasformazione lineare


# Log-densità posteriori proporzionale
def log_posterior_rho(rho, u, v, alpha, beta):
    """Log-densità posteriori proporzionale per rho."""
    log_prior = log_prior_rho(rho, alpha, beta)
    if log_prior == -np.inf:
        return -np.inf

    log_like = gaussian_copula_loglik(rho, u, v)
    return log_prior + log_like


# Algoritmo Metropolis-Hastings per campionamento dalla posteriori
def metropolis_hastings(log_posterior_func, initial_value, n_samples, proposal_width, **kwargs):
    """
    Implementazione dell'algoritmo Metropolis-Hastings per il campionamento dalla distribuzione posteriori.

    Args:
        log_posterior_func: Funzione che calcola il logaritmo della densità posteriori proporzionale
        initial_value: Valore iniziale del parametro
        n_samples: Numero di campioni da generare
        proposal_width: Ampiezza della distribuzione di proposta (normale)
        **kwargs: Argomenti aggiuntivi da passare a log_posterior_func

    Returns:
        Array di campioni dalla distribuzione posteriori
    """
    samples = np.zeros(n_samples)
    samples[0] = initial_value

    # Valore corrente del log-posteriori
    current_log_prob = log_posterior_func(initial_value, **kwargs)

    # Contatore delle accettazioni
    n_accepted = 0

    for i in range(1, n_samples):
        # Proposta di un nuovo valore
        proposal = samples[i - 1] + np.random.normal(0, proposal_width)

        # Calcolo del log-posteriori per il valore proposto
        proposal_log_prob = log_posterior_func(proposal, **kwargs)

        # Rapporto di accettazione (log-scala)
        log_acceptance_ratio = proposal_log_prob - current_log_prob

        # Decisione di accettazione
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            samples[i] = proposal
            current_log_prob = proposal_log_prob
            n_accepted += 1
        else:
            samples[i] = samples[i - 1]

    acceptance_rate = n_accepted / (n_samples - 1)
    return samples, acceptance_rate


# Esecuzione del MCMC per la copula Gaussiana
print("\n5.1 MCMC per la copula Gaussiana")

# Riduce il dataset per velocizzare il MCMC
# Prendiamo un campione casuale per velocizzare l'inferenza
n_data_samples = min(1000, len(u))  # Massimo 1000 campioni
idx = np.random.choice(range(len(u)), n_data_samples, replace=False)
u_sample = u[idx]
v_sample = v[idx]

print(f"Utilizziamo {n_data_samples} campioni casuali per l'inferenza bayesiana")

# Definizione dei parametri della prior
alpha_prior = 1 + 5 * abs(pearson_corr)  # Basato sulla correlazione empirica
beta_prior = 1 + 5 * (1 - abs(pearson_corr))
print(f"Prior: Beta({alpha_prior:.2f}, {beta_prior:.2f}) trasformata in [-1, 1]")

# Parametri del MCMC
n_samples_mcmc = 10000
proposal_width = 0.05
burn_in = 2000

# Esecuzione del MCMC
initial_rho = pearson_corr
samples_rho, acceptance_rate = metropolis_hastings(
    log_posterior_rho,
    initial_rho,
    n_samples_mcmc,
    proposal_width,
    u=u_sample,
    v=v_sample,
    alpha=alpha_prior,
    beta=beta_prior
)

print(f"MCMC completato, tasso di accettazione: {acceptance_rate:.4f}")

# Rimuovi burn-in e analizza i risultati
samples_rho_final = samples_rho[burn_in:]
rho_mean = np.mean(samples_rho_final)
rho_median = np.median(samples_rho_final)
rho_ci = np.percentile(samples_rho_final, [2.5, 97.5])

print(f"\nRisultati per la copula Gaussiana:")
print(f"ρ (media posteriori): {rho_mean:.4f}")
print(f"ρ (mediana posteriori): {rho_median:.4f}")
print(f"ρ (intervallo credibile 95%): [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}]")

# Visualizzazione della distribuzione posteriori
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(range(len(samples_rho_final)), samples_rho_final, alpha=0.5)
plt.axhline(y=rho_mean, color='r', linestyle='-', label=f'Media ({rho_mean:.4f})')
plt.axhline(y=rho_ci[0], color='g', linestyle='--', label=f'2.5% ({rho_ci[0]:.4f})')
plt.axhline(y=rho_ci[1], color='g', linestyle='--', label=f'97.5% ({rho_ci[1]:.4f})')
plt.title('Trace plot dei campioni MCMC (dopo burn-in)')
plt.xlabel('Iterazione')
plt.ylabel('ρ')
plt.legend()

plt.subplot(2, 1, 2)
plt.hist(samples_rho_final, bins=50, alpha=0.7, density=True)
plt.axvline(x=rho_mean, color='r', linestyle='-', label=f'Media ({rho_mean:.4f})')
plt.axvline(x=rho_median, color='b', linestyle='--', label=f'Mediana ({rho_median:.4f})')
plt.axvline(x=rho_ci[0], color='g', linestyle='--', label=f'2.5% ({rho_ci[0]:.4f})')
plt.axvline(x=rho_ci[1], color='g', linestyle='--', label=f'97.5% ({rho_ci[1]:.4f})')
plt.title('Distribuzione posteriori di ρ (Copula Gaussiana)')
plt.xlabel('ρ')
plt.ylabel('Densità')
plt.legend()

plt.tight_layout()
plt.savefig('gaussian_posterior_mcmc.png')
print("✅ Grafico della distribuzione posteriori salvato come 'gaussian_posterior_mcmc.png'")

# Confronto con la stima MLE
from scipy.optimize import minimize


def neg_gaussian_loglik(rho, u, v):
    """Log-verosimiglianza negativa per minimizzazione."""
    if abs(rho[0]) >= 0.999:
        return np.inf

    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)

    term1 = -0.5 * np.log(1 - rho[0] ** 2)
    term2 = -0.5 * (rho[0] ** 2 * (x ** 2 + y ** 2) - 2 * rho[0] * x * y) / (1 - rho[0] ** 2)

    return -np.sum(term1 + term2)


result_mle = minimize(neg_gaussian_loglik, [pearson_corr],
                      args=(u_sample, v_sample),
                      bounds=[(-0.999, 0.999)])
rho_mle = result_mle.x[0]

print(f"\nConfronto tra stime:")
print(f"ρ (correlazione di Pearson): {pearson_corr:.4f}")
print(f"ρ (MLE): {rho_mle:.4f}")
print(f"ρ (Bayesiano): {rho_mean:.4f}")

# 6. Simulazione di dati dalla copula stimata
print("\n6️⃣ Simulazione dalla copula stimata")
print("-" * 50)

# Generiamo punti dalla copula Gaussiana con il parametro stimato
n_points = 1000
mean = [0, 0]
cov = [[1, rho_mean], [rho_mean, 1]]
z = np.random.multivariate_normal(mean, cov, n_points)
simulated_data = stats.norm.cdf(z)

# Visualizzazione dei dati originali e simulati
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.scatter(u, v, alpha=0.5, s=10)
plt.title('Dati originali uniformati')
plt.xlabel('Open (uniforme)')
plt.ylabel('Close (uniforme)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(simulated_data[:, 0], simulated_data[:, 1], alpha=0.5, s=10, c='red')
plt.title(f'Copula Gaussiana stimata (ρ = {rho_mean:.4f})')
plt.xlabel('U1')
plt.ylabel('U2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulated_vs_original.png')
print("✅ Grafico di confronto salvato come 'simulated_vs_original.png'")

print("\n✅ Analisi bayesiana completata con successo!")