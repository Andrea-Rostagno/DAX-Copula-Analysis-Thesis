# Implementazione di stimatori MCMC per diverse copule sui dati DAX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import warnings

warnings.filterwarnings('ignore')

print("📊 STIMA BAYESIANA DELLE COPULE CON MCMC - DATI DAX")
print("=" * 70)

# 0. Caricamento e preparazione dei dati DAX
print("\n0️⃣ Caricamento dati DAX")
print("-" * 50)

try:
    # Carica il file DAX pulito
    df = pd.read_csv('DAX_cleaned.csv', index_col=0, parse_dates=True)
    print("✅ Dati DAX caricati correttamente")
    print(f"Dimensioni del dataset: {df.shape}")
    print(f"Periodo: {df.index.min()} - {df.index.max()}")
    print(f"Colonne: {df.columns.tolist()}")
    print("\nPrime 5 righe:")
    print(df.head())

    # Identificazione delle colonne di prezzo
    price_cols = [col for col in df.columns if col in
                  ['Open', 'Close']]#, 'High', 'Low', 'Adj Close', 'Price', 'High', 'Low', 'Adj Close', 'Price'

    if len(price_cols) >= 2:
        col1, col2 = price_cols[:2]
    else:
        # Usa le prime due colonne numeriche disponibili
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[:2]
        else:
            raise ValueError("Non sufficienti colonne numeriche nel dataset")

    print(f"\nUtilizzo delle colonne: '{col1}' e '{col2}'")

    # Calcolo dei rendimenti logaritmici
    returns = pd.DataFrame({
        'r1': np.log(df[col1] / df[col1].shift(1)),
        'r2': np.log(df[col2] / df[col2].shift(1))
    }).dropna()

    print(f"Calcolati {len(returns)} rendimenti logaritmici")

    # Calcolo della volatilità rolling
    print("\nCalcolo della volatilità rolling...")
    window = 21  # Finestra di 21 giorni (circa un mese di trading)

    # Per dataset molto grandi, campionamento per migliorare le prestazioni
    if len(returns) > 50000:
        print(f"Dataset molto grande, campionamento 1 ogni {len(returns) // 50000 + 1} righe...")
        sample_step = len(returns) // 50000 + 1
        returns_sample = returns.iloc[::sample_step].copy()
        print(f"Dataset ridotto da {len(returns)} a {len(returns_sample)} righe")
    else:
        returns_sample = returns.copy()

    # Calcolo della volatilità
    returns_sample['vol1'] = returns_sample['r1'].rolling(window=window).std() * np.sqrt(252)  # Annualizzata
    returns_sample['vol2'] = returns_sample['r2'].rolling(window=window).std() * np.sqrt(252)  # Annualizzata

    # Rimozione dei NaN
    vol_returns = returns_sample.dropna()
    print(f"Volatilità calcolata su {len(vol_returns)} punti")

    # Trasformazione in distribuzioni uniformi per l'analisi delle copule
    print("\nTrasformazione in distribuzioni uniformi...")
    u1 = stats.rankdata(vol_returns['vol1']) / (len(vol_returns) + 1)
    u2 = stats.rankdata(vol_returns['vol2']) / (len(vol_returns) + 1)

    # Calcolo delle misure di dipendenza
    print("\nMisure di dipendenza tra le volatilità:")
    pearson_corr = np.corrcoef(vol_returns['vol1'], vol_returns['vol2'])[0, 1]
    kendall_tau, _ = stats.kendalltau(vol_returns['vol1'], vol_returns['vol2'])
    spearman_rho, _ = stats.spearmanr(vol_returns['vol1'], vol_returns['vol2'])

    print(f"Correlazione di Pearson: {pearson_corr:.4f}")
    print(f"Tau di Kendall: {kendall_tau:.4f}")
    print(f"Rho di Spearman: {spearman_rho:.4f}")

    # Visualizzazione dei dati uniformati
    plt.figure(figsize=(10, 8))
    plt.scatter(u1, u2, alpha=0.7)
    plt.title('Scatter Plot delle Volatilità Uniformate')
    plt.xlabel(f'{col1} Volatilità (Uniforme)')
    plt.ylabel(f'{col2} Volatilità (Uniforme)')
    plt.grid(True, alpha=0.3)
    plt.savefig('dax_uniform_volatility.png', dpi=300)
    print("✅ Grafico delle volatilità uniformate salvato come 'dax_uniform_volatility.png'")

except FileNotFoundError:
    print("❌ File 'DAX_cleaned.csv' non trovato!")
    raise


# 1. Definizione delle distribuzioni di copule per PyMC3

# 1.1 Copula Gaussiana
def gaussian_copula_logp(u, v, rho):
    """Log densità della copula Gaussiana"""
    norm_u = stats.norm.ppf(u)
    norm_v = stats.norm.ppf(v)

    # Evita valori estremi che potrebbero causare problemi numerici
    norm_u = tt.clip(norm_u, -8, 8)
    norm_v = tt.clip(norm_v, -8, 8)

    term1 = -0.5 * tt.log(1 - rho ** 2)
    term2 = -1 / (2 * (1 - rho ** 2)) * (norm_u ** 2 + norm_v ** 2 - 2 * rho * norm_u * norm_v)
    term3 = 0.5 * (norm_u ** 2 + norm_v ** 2)

    return term1 + term2 + term3


# 1.2 Copula t-Student
class StudentTCopula(pm.Continuous):
    """Distribuzione della copula t-Student"""

    def __init__(self, rho, nu, u=None, v=None, *args, **kwargs):
        self.rho = rho
        self.nu = nu
        self.u = u
        self.v = v
        super().__init__(*args, **kwargs)

    def logp(self, x):
        """Log densità della copula t-Student"""
        u = self.u
        v = self.v
        rho = self.rho
        nu = self.nu

        t_u = stats.t.ppf(u, nu)
        t_v = stats.t.ppf(v, nu)

        # Evita valori estremi
        t_u = tt.clip(t_u, -8, 8)
        t_v = tt.clip(t_v, -8, 8)

        w = (t_u ** 2 + t_v ** 2 - 2 * rho * t_u * t_v) / (1 - rho ** 2)

        term1 = tt.log(tt.gamma((nu + 2) / 2)) + tt.log(tt.gamma(nu / 2)) - 2 * tt.log(tt.gamma((nu + 1) / 2))
        term2 = -0.5 * tt.log(1 - rho ** 2)
        term3 = -(nu + 2) / 2 * tt.log(1 + w / nu)
        term4 = -(nu + 1) / 2 * tt.log(1 + t_u ** 2 / nu) - (nu + 1) / 2 * tt.log(1 + t_v ** 2 / nu)

        return term1 + term2 + term3 - term4


# 1.3 Copula di Clayton
def clayton_copula_logp(u, v, theta):
    """Log densità della copula di Clayton"""
    # Evita problemi numerici per valori molto piccoli
    u = tt.clip(u, 0.001, 0.999)
    v = tt.clip(v, 0.001, 0.999)

    term1 = tt.log(1 + theta)
    term2 = -(1 + theta) * (tt.log(u) + tt.log(v))
    term3 = -(2 + 1 / theta) * tt.log(u ** (-theta) + v ** (-theta) - 1)

    return term1 + term2 + term3


# 1.4 Copula di Gumbel
def gumbel_copula_logp(u, v, theta):
    """Log densità della copula di Gumbel"""
    # Evita problemi numerici per valori molto piccoli
    u = tt.clip(u, 0.001, 0.999)
    v = tt.clip(v, 0.001, 0.999)

    log_u = -tt.log(u)
    log_v = -tt.log(v)

    w = (log_u ** theta + log_v ** theta) ** (1 / theta)

    term1 = -w
    term2 = tt.log(w * (log_u * log_v) ** (theta - 1))
    term3 = tt.log(theta - 1 + w) - tt.log(u * v)

    return term1 + term2 + term3


# 1.5 Copula di Frank
def frank_copula_logp(u, v, theta):
    """Log densità della copula di Frank"""
    # Evita problemi numerici per valori molto piccoli
    u = tt.clip(u, 0.001, 0.999)
    v = tt.clip(v, 0.001, 0.999)

    # Gestione dei casi limite per theta
    eps = 1e-10
    theta = tt.switch(tt.abs_(theta) < eps, eps, theta)

    term1 = tt.log(tt.abs_(theta)) - tt.log(1 - tt.exp(-tt.abs_(theta)))
    term2 = -theta * (u + v)
    term3 = -2 * tt.log(1 + tt.exp(-theta) * (tt.exp(-theta * u) - 1) * (tt.exp(-theta * v) - 1))

    return term1 + term2 + term3


# 2. Funzione per la stima MCMC delle copule
def fit_copula_mcmc(u, v, copula_type='gaussian', samples=2000, tune=1000, chains=2, cores=1):
    """
    Stima i parametri di una copula usando MCMC

    Parametri:
    u, v (array): Dati uniformi
    copula_type (str): Tipo di copula ('gaussian', 't', 'clayton', 'gumbel', 'frank')
    samples (int): Numero di campioni MCMC
    tune (int): Numero di step di tuning
    chains (int): Numero di catene
    cores (int): Numero di core da utilizzare

    Returns:
    dict: Risultati della stima con trace, summary e parametri
    """
    print(f"\n🔄 Iniziando stima MCMC per copula {copula_type}...")

    with pm.Model() as model:
        # Priors
        if copula_type == 'gaussian':
            # Parametro di correlazione tra -1 e 1
            rho = pm.Uniform('rho', -0.99, 0.99)

            # Likelihood
            pm.DensityDist('likelihood', lambda x: gaussian_copula_logp(u, v, rho), observed=0)

        elif copula_type == 't':
            # Parametri: correlazione e gradi di libertà
            rho = pm.Uniform('rho', -0.99, 0.99)
            nu = pm.Gamma('nu', alpha=2, beta=0.1)  # Prior per i gradi di libertà

            # Likelihood
            StudentTCopula('likelihood', rho=rho, nu=nu, u=u, v=v, observed=0)

        elif copula_type == 'clayton':
            # Parametro theta > 0
            theta = pm.Gamma('theta', alpha=1, beta=1)

            # Likelihood
            pm.DensityDist('likelihood', lambda x: clayton_copula_logp(u, v, theta), observed=0)

        elif copula_type == 'gumbel':
            # Parametro theta >= 1
            # Usiamo un offset per garantire theta >= 1
            theta_offset = pm.Exponential('theta_offset', 1)
            theta = pm.Deterministic('theta', 1 + theta_offset)

            # Likelihood
            pm.DensityDist('likelihood', lambda x: gumbel_copula_logp(u, v, theta), observed=0)

        elif copula_type == 'frank':
            # Parametro theta può essere positivo o negativo
            theta = pm.Normal('theta', mu=0, sd=5)

            # Likelihood
            pm.DensityDist('likelihood', lambda x: frank_copula_logp(u, v, theta), observed=0)

        else:
            raise ValueError(f"Tipo di copula non supportato: {copula_type}")

        # Sampling
        print(f"  Esecuzione sampling con {chains} catene, {samples} campioni, {tune} passi di tuning...")
        trace = pm.sample(samples, tune=tune, chains=chains, cores=cores,
                          return_inferencedata=True)

    # Calcola la media posteriore dei parametri
    print("  Calcolo statistiche posteriori...")
    summary = az.summary(trace)

    result = {
        'trace': trace,
        'summary': summary,
        'parameters': {param: summary.loc[param, 'mean'] for param in summary.index}
    }

    print(f"✅ Stima completata per copula {copula_type}")
    print(f"   Parametri stimati: {result['parameters']}")

    return result


# 3. Funzione per confrontare tutte le copule con MCMC
def compare_copulas_mcmc(u, v, samples=2000, tune=1000, chains=2, cores=1):
    """
    Stima e confronta diverse copule usando MCMC

    Parametri:
    u, v (array): Dati uniformi
    samples, tune, chains, cores: Parametri per MCMC

    Returns:
    dict: Risultati per ciascuna copula
    DataFrame: Confronto dei modelli
    """
    print("\n2️⃣ Avvio stima MCMC per tutte le copule")
    print("-" * 50)
    print("⚠️ Questo processo potrebbe richiedere tempo...")

    results = {}
    copula_types = ['gaussian', 't', 'clayton', 'gumbel', 'frank']

    for copula in copula_types:
        try:
            results[copula] = fit_copula_mcmc(u, v, copula_type=copula,
                                              samples=samples, tune=tune,
                                              chains=chains, cores=cores)
        except Exception as e:
            print(f"❌ Errore nella stima della copula {copula}: {e}")

    # Calcolo delle metriche di confronto dei modelli
    print("\n3️⃣ Confronto dei modelli")
    print("-" * 50)

    model_comparison = []

    for copula, result in results.items():
        try:
            waic = az.waic(result['trace'])
            loo = az.loo(result['trace'])

            model_info = {
                'Copula': copula,
                'WAIC': waic.waic,
                'WAIC_SE': waic.waic_se,
                'LOO': loo.loo,
                'LOO_SE': loo.loo_se,
                'Parametri': str(result['parameters'])
            }
            model_comparison.append(model_info)
        except Exception as e:
            print(f"❌ Errore nel calcolo delle metriche per copula {copula}: {e}")

    model_comparison_df = pd.DataFrame(model_comparison).sort_values('WAIC')
    print("\nRisultati del confronto dei modelli (ordinati per WAIC crescente):")
    print(model_comparison_df)

    # Salva i risultati
    model_comparison_df.to_csv('mcmc_copula_comparison_results.csv', index=False)
    print("✅ Risultati salvati in 'mcmc_copula_comparison_results.csv'")

    return results, model_comparison_df


# 4. Funzione per simulare da copule con parametri stimati
def simulate_from_copula(copula_type, parameters, n_sim=1000):
    """
    Simula dati da una copula con parametri specificati

    Parametri:
    copula_type (str): Tipo di copula
    parameters (dict): Parametri stimati
    n_sim (int): Numero di simulazioni

    Returns:
    array: Dati simulati (u, v)
    """
    print(f"\nSimulazione dalla copula {copula_type} con parametri: {parameters}")

    if copula_type == 'gaussian':
        rho = parameters['rho']
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        z = np.random.multivariate_normal(mean, cov, n_sim)
        simulated_uniform = stats.norm.cdf(z)

    elif copula_type == 't':
        rho = parameters['rho']
        nu = parameters['nu']
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        z = np.random.multivariate_normal(mean, cov, n_sim)

        # Genera gradi di libertà per t
        w = np.random.chisquare(nu, n_sim) / nu
        z = z / np.sqrt(w)[:, np.newaxis]

        simulated_uniform = stats.t.cdf(z, nu)

    elif copula_type == 'clayton':
        theta = parameters['theta']
        v1 = np.random.uniform(0, 1, n_sim)
        v2 = np.random.uniform(0, 1, n_sim)

        u1_sim = v1
        u2_sim = (1 + (v1 ** (-theta) - 1) / (v2 ** (-1 / theta) - 1)) ** (-1 / theta)

        simulated_uniform = np.column_stack((u1_sim, u2_sim))

    elif copula_type == 'gumbel':
        theta = parameters['theta']
        # Genera dalla copula di Gumbel (approssimazione)
        v1 = np.random.uniform(0, 1, n_sim)
        v2 = np.random.uniform(0, 1, n_sim)

        from scipy.stats import genextreme
        gamma = -1 / theta
        z = genextreme.rvs(gamma, size=n_sim)
        u1_sim = np.exp(-np.exp(-z))
        u2_sim = np.exp(-np.exp(-z - np.log(v2)))

        simulated_uniform = np.column_stack((u1_sim, u2_sim))

    elif copula_type == 'frank':
        theta = parameters['theta']
        v1 = np.random.uniform(0, 1, n_sim)
        v2 = np.random.uniform(0, 1, n_sim)

        u1_sim = v1
        if abs(theta) < 1e-10:
            u2_sim = v2  # Indipendenza
        else:
            u2_sim = -np.log(1 + (v2 * (1 - np.exp(-theta))) /
                             (np.exp(-theta * v1) * (np.exp(-theta) - 1) + (1 - np.exp(-theta)))) / theta

        simulated_uniform = np.column_stack((u1_sim, u2_sim))

    else:
        raise ValueError(f"Tipo di copula non supportato: {copula_type}")

    return simulated_uniform


# 5. Funzione per visualizzare i risultati
def plot_copula_results(u, v, results, model_comparison, n_sim=1000):
    """
    Visualizza i risultati delle stime MCMC e le simulazioni

    Parametri:
    u, v (array): Dati originali
    results (dict): Risultati delle stime MCMC
    model_comparison (DataFrame): Confronto dei modelli
    n_sim (int): Numero di simulazioni
    """
    print("\n4️⃣ Visualizzazione dei risultati")
    print("-" * 50)

    # Determina la copula migliore
    best_copula = model_comparison.iloc[0]['Copula']
    print(f"\n🏆 La copula migliore secondo WAIC è: {best_copula}")

    # Visualizza i dati originali e le simulazioni dalla migliore copula
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(u, v, alpha=0.7)
    plt.title('Dati Originali Uniformati')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    simulated = simulate_from_copula(best_copula, results[best_copula]['parameters'], n_sim)
    plt.scatter(simulated[:, 0], simulated[:, 1], alpha=0.7, color='red')
    plt.title(f'Copula {best_copula} Simulata\n{str(results[best_copula]["parameters"])}')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcmc_copula_comparison.png', dpi=300)
    print("✅ Grafico di confronto salvato come 'mcmc_copula_comparison.png'")

    # Visualizza le tracce posteriori per la migliore copula
    az.plot_trace(results[best_copula]['trace'])
    plt.savefig('mcmc_best_trace.png', dpi=300)
    print("✅ Grafico delle tracce salvato come 'mcmc_best_trace.png'")

    # Visualizza la densità posteriore per la migliore copula
    az.plot_posterior(results[best_copula]['trace'])
    plt.savefig('mcmc_best_posterior.png', dpi=300)
    print("✅ Grafico della densità posteriore salvato come 'mcmc_best_posterior.png'")

    # Visualizza il confronto dei modelli
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    waic_values = model_comparison['WAIC'].values
    copulas = model_comparison['Copula'].values

    # Normalizza i valori WAIC per la visualizzazione
    waic_norm = (waic_values - np.min(waic_values)) / (np.max(waic_values) - np.min(waic_values))

    # Crea il grafico a barre
    bars = ax.bar(copulas, -waic_norm, alpha=0.7)

    # Formatta il grafico
    ax.set_ylabel('WAIC Score Normalizzato (più alto è migliore)')
    ax.set_title('Confronto dei Modelli di Copula con MCMC')
    ax.set_ylim(-1, 0.1)

    # Aggiungi etichette con i valori WAIC
    for i, (bar, waic) in enumerate(zip(bars, waic_values)):
        ax.text(i, -0.05, f"{waic:.1f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('mcmc_model_comparison.png', dpi=300)
    print("✅ Grafico del confronto dei modelli salvato come 'mcmc_model_comparison.png'")


# 6. Confronto con le stime di massima verosimiglianza
def compare_with_mle(u, v, mcmc_results, kendall_tau):
    """
    Confronta i risultati MCMC con le stime di massima verosimiglianza

    Parametri:
    u, v (array): Dati uniformi
    mcmc_results (dict): Risultati delle stime MCMC
    kendall_tau (float): Tau di Kendall stimato dai dati
    """
    print("\n5️⃣ Confronto con stime di massima verosimiglianza")
    print("-" * 50)

    from scipy.optimize import minimize

    # Funzioni negative log-likelihood per la massima verosimiglianza
    def neg_gaussian_loglik(rho, u, v):
        if isinstance(rho, np.ndarray):
            rho = rho[0]
        return -np.sum(gaussian_copula_logp(u, v, rho).eval())

    def neg_t_loglik(params, u, v):
        rho, nu = params
        if nu <= 2:
            return 1e10  # Penalità per valori non validi
        # Usa una classe personalizzata per la valutazione
        model = StudentTCopula('temp', rho=rho, nu=nu, u=u, v=v, observed=0)
        return -model.logp(0).eval()

    def neg_clayton_loglik(theta, u, v):
        if isinstance(theta, np.ndarray):
            theta = theta[0]
        if theta <= 0:
            return 1e10
        return -np.sum(clayton_copula_logp(u, v, theta).eval())

    def neg_gumbel_loglik(theta, u, v):
        if isinstance(theta, np.ndarray):
            theta = theta[0]
        if theta < 1:
            return 1e10
        return -np.sum(gumbel_copula_logp(u, v, theta).eval())

    def neg_frank_loglik(theta, u, v):
        if isinstance(theta, np.ndarray):
            theta = theta[0]
        return -np.sum(frank_copula_logp(u, v, theta).eval())

    # Stima MLE per copula Gaussiana
    init_rho = np.corrcoef(u, v)[0, 1]
    bounds_gaussian = [(-0.999, 0.999)]
    result_gaussian = minimize(neg_gaussian_loglik, [init_rho], args=(u, v),
                               bounds=bounds_gaussian, method='L-BFGS-B')

    # Stima MLE per copula t-Student
    init_params_t = [init_rho, 5]
    bounds_t = [(-0.999, 0.999), (2.001, 30)]
    result_t = minimize(neg_t_loglik, init_params_t, args=(u, v),
                        bounds=bounds_t, method='L-BFGS-B')

    # Stima MLE per copula Clayton (se applicabile)
    if kendall_tau > 0:
        init_theta_clayton = 2 * kendall_tau / (1 - kendall_tau)
        bounds_clayton = [(0.001, 20)]
        result_clayton = minimize(neg_clayton_loglik, [init_theta_clayton], args=(u, v),
                                  bounds=bounds_clayton, method='L-BFGS-B')
    else:
        result_clayton = None

    # Stima MLE per copula Gumbel (se applicabile)
    if kendall_tau > 0:
        init_theta_gumbel = 1 / (1 - kendall_tau)
        bounds_gumbel = [(1.001, 20)]
        result_gumbel = minimize(neg_gumbel_loglik, [init_theta_gumbel], args=(u, v),
                                 bounds=bounds_gumbel, method='L-BFGS-B')
    else:
        result_gumbel = None

    # Stima MLE per copula Frank
    init_theta_frank = 0
    bounds_frank = [(-20, 20)]
    result_frank = minimize(neg_frank_loglik, [init_theta_frank], args=(u, v),
                            bounds=bounds_frank, method='L-BFGS-B')

    # Crea DataFrame per il confronto
    comparison_data = []

    # Gaussiana
    gaussian_mcmc = mcmc_results.get('gaussian', {}).get('parameters', {}).get('rho', np.nan)
    comparison_data.append({
        'Copula': 'Gaussiana',
        'Parametro': 'rho',
        'MLE': result_gaussian.x[0],
        'MCMC': gaussian_mcmc,
        'Differenza %': np.abs(result_gaussian.x[0] - gaussian_mcmc) / np.abs(
            result_gaussian.x[0]) * 100 if not np.isnan(gaussian_mcmc) else np.nan
    })

    # t-Student
    t_rho_mcmc = mcmc_results.get('t', {}).get('parameters', {}).get('rho', np.nan)
    t_nu_mcmc = mcmc_results.get('t', {}).get('parameters', {}).get('nu', np.nan)

    if not np.isnan(t_rho_mcmc):
        comparison_data.append({
            'Copula': 't-Student',
            'Parametro': 'rho',
            'MLE': result_t.x[0],
            'MCMC': t_rho_mcmc,
            'Differenza %': np.abs(result_t.x[0] - t_rho_mcmc) / np.abs(result_t.x[0]) * 100
        })

    if not np.isnan(t_nu_mcmc):
        comparison_data.append({
            'Copula': 't-Student',
            'Parametro': 'nu',
            'MLE': result_t.x[1],
            'MCMC': t_nu_mcmc,
            'Differenza %': np.abs(result_t.x[1] - t_nu_mcmc) / np.abs(result_t.x[1]) * 100
        })

    # Clayton
    if result_clayton is not None:
        clayton_mcmc = mcmc_results.get('clayton', {}).get('parameters', {}).get('theta', np.nan)
        if not np.isnan(clayton_mcmc):
            comparison_data.append({
                'Copula': 'Clayton',
                'Parametro': 'theta',
                'MLE': result_clayton.x[0],
                'MCMC': clayton_mcmc,
                'Differenza %': np.abs(result_clayton.x[0] - clayton_mcmc) / np.abs(
                    result_clayton.x[0]) * 100 if not np.isnan(clayton_mcmc) else np.nan
            })

    # Gumbel
    if result_gumbel is not None:
        gumbel_mcmc = mcmc_results.get('gumbel', {}).get('parameters', {}).get('theta', np.nan)
        if not np.isnan(gumbel_mcmc):
            comparison_data.append({
                'Copula': 'Gumbel',
                'Parametro': 'theta',
                'MLE': result_gumbel.x[0],
                'MCMC': gumbel_mcmc,
                'Differenza %': np.abs(result_gumbel.x[0] - gumbel_mcmc) / np.abs(
                    result_gumbel.x[0]) * 100 if not np.isnan(gumbel_mcmc) else np.nan
            })

    # Frank
    frank_mcmc = mcmc_results.get('frank', {}).get('parameters', {}).get('theta', np.nan)
    if not np.isnan(frank_mcmc):
        comparison_data.append({
            'Copula': 'Frank',
            'Parametro': 'theta',
            'MLE': result_frank.x[0],
            'MCMC': frank_mcmc,
            'Differenza %': np.abs(result_frank.x[0] - frank_mcmc) / np.abs(result_frank.x[0]) * 100 if not np.isnan(
                frank_mcmc) else np.nan
        })

    # Crea DataFrame e lo visualizza
    comparison_df = pd.DataFrame(comparison_data)
    print("\nConfronto tra stime MLE e MCMC:")
    print(comparison_df)

    # Salva i risultati
    comparison_df.to_csv('mle_mcmc_comparison.csv', index=False)
    print("✅ Confronto MLE vs MCMC salvato in 'mle_mcmc_comparison.csv'")

    # Visualizza il confronto
    plt.figure(figsize=(12, 8))

    # Filtra i parametri per tipo
    rho_data = comparison_df[comparison_df['Parametro'] == 'rho']
    theta_data = comparison_df[comparison_df['Parametro'] == 'theta']
    nu_data = comparison_df[comparison_df['Parametro'] == 'nu']

    # Crea subplot per ogni tipo di parametro
    n_plots = sum([len(rho_data) > 0, len(theta_data) > 0, len(nu_data) > 0])
    plot_idx = 1

    if len(rho_data) > 0:
        plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1

        labels = rho_data['Copula'].values
        mle_values = rho_data['MLE'].values
        mcmc_values = rho_data['MCMC'].values

        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width / 2, mle_values, width, label='MLE')
        plt.bar(x + width / 2, mcmc_values, width, label='MCMC')

        plt.ylabel('Valore rho')
        plt.title('Confronto parametro rho')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(alpha=0.3)

    if len(theta_data) > 0:
        plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1

        labels = theta_data['Copula'].values
        mle_values = theta_data['MLE'].values
        mcmc_values = theta_data['MCMC'].values

        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width / 2, mle_values, width, label='MLE')
        plt.bar(x + width / 2, mcmc_values, width, label='MCMC')

        plt.ylabel('Valore theta')
        plt.title('Confronto parametro theta')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(alpha=0.3)

    if len(nu_data) > 0:
        plt.subplot(n_plots, 1, plot_idx)

        labels = nu_data['Copula'].values
        mle_values = nu_data['MLE'].values
        mcmc_values = nu_data['MCMC'].values

        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width / 2, mle_values, width, label='MLE')
        plt.bar(x + width / 2, mcmc_values, width, label='MCMC')

        plt.ylabel('Valore nu')
        plt.title('Confronto parametro nu')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('mle_mcmc_comparison.png', dpi=300)
    print("✅ Grafico confronto MLE vs MCMC salvato come 'mle_mcmc_comparison.png'")

    return comparison_df


# 7. Esecuzione principale
def main(sample_size=None, mcmc_samples=2000, mcmc_tune=1000, mcmc_chains=2, mcmc_cores=1):
    """
    Funzione principale che esegue l'intera analisi MCMC delle copule

    Parametri:
    sample_size (int): Dimensione del campione da utilizzare, None per usare tutti i dati
    mcmc_samples (int): Numero di campioni MCMC
    mcmc_tune (int): Numero di passi di tuning
    mcmc_chains (int): Numero di catene
    mcmc_cores (int): Numero di core da utilizzare
    """
    global u1, u2  # Usa le variabili globali u1 e u2

    print("\n1️⃣ Preparazione dei dati per MCMC")
    print("-" * 50)

    # Limita la dimensione del campione se specificato
    if sample_size is not None and sample_size < len(u1):
        print(f"⚠️ Campionamento dati: utilizzando {sample_size} punti dei {len(u1)} disponibili")
        indices = np.random.choice(len(u1), sample_size, replace=False)
        u1_sample = u1[indices]
        u2_sample = u2[indices]
    else:
        u1_sample = u1
        u2_sample = u2

    print(f"Dimensione campione utilizzato: {len(u1_sample)}")

    # Esegui la stima MCMC per tutte le copule
    results, model_comparison = compare_copulas_mcmc(
        u1_sample, u2_sample,
        samples=mcmc_samples,
        tune=mcmc_tune,
        chains=mcmc_chains,
        cores=mcmc_cores
    )

    # Visualizza i risultati
    plot_copula_results(u1_sample, u2_sample, results, model_comparison)

    # Confronta con le stime MLE
    comparison_df = compare_with_mle(u1_sample, u2_sample, results, kendall_tau)

    print("\n✅ Analisi MCMC completata con successo!")

    return results, model_comparison, comparison_df


# Esegui lo script se chiamato direttamente

if __name__ == "__main__":
    # Parametri configurabili
    SAMPLE_SIZE = 5000  # Imposta a None per utilizzare tutti i dati
    MCMC_SAMPLES = 2000  # Numero di campioni MCMC
    MCMC_TUNE = 1000  # Numero di passi di tuning
    MCMC_CHAINS = 2  # Numero di catene
    MCMC_CORES = 1  # Numero di core (impostare a più di 1 solo se multiprocessing disponibile)

    print("\n⚙️ Configurazione:")
    print(f"  Dimensione campione: {SAMPLE_SIZE if SAMPLE_SIZE is not None else 'Tutti i dati'}")
    print(f"  Campioni MCMC: {MCMC_SAMPLES}")
    print(f"  Passi tuning: {MCMC_TUNE}")
    print(f"  Catene: {MCMC_CHAINS}")
    print(f"  Core: {MCMC_CORES}")

    results, model_comparison, comparison_df = main(
        sample_size=SAMPLE_SIZE,
        mcmc_samples=MCMC_SAMPLES,
        mcmc_tune=MCMC_TUNE,
        mcmc_chains=MCMC_CHAINS,
        mcmc_cores=MCMC_CORES
    )
