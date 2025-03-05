# Wrappa l'intero script in un blocco try/except
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from scipy.optimize import minimize
    import warnings
    from matplotlib.dates import date2num
    import datetime

    warnings.filterwarnings('ignore')

    print("📊 SELEZIONE DELLE COPULE E ANALISI DI VOLATILITÀ")
    print("=" * 70)

    # 1. Caricamento e preprocessing dei dati
    print("\n1️⃣ Caricamento e preprocessing dei dati")
    print("-" * 50)

    try:
        # Carica il file DAX, se disponibile
        df = pd.read_csv('DAX_cleaned.csv', index_col=0, parse_dates=True)
        print("✅ Caricato file pulito 'DAX_cleaned.csv'")
    except FileNotFoundError:
        try:
            print("⚠️ File pulito non trovato, provo con il file originale...")
            df = pd.read_csv('DAX_3Y-1M.csv',
                             sep=None,
                             engine='python',
                             encoding='utf-8',
                             on_bad_lines='skip',
                             na_values=['na', 'NA', 'N/A', ''])

            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                print(f"✅ Usando '{date_cols[0]}' come colonna datetime")
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                df.set_index(date_cols[0], inplace=True)

            print("✅ Caricamento completato con opzioni robuste")
        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            print("⚠️ Utilizzo di un dataset di esempio...")

            # Creazione dataset di esempio
            np.random.seed(42)
            n = 1000
            dates = pd.date_range(start='2020-01-01', periods=n, freq='D')

            # Simulazione di dati con due asset correlati
            rho = 0.7
            cov = [[1.0, rho], [rho, 1.0]]
            returns = np.random.multivariate_normal([0.0005, 0.0003],
                                                    [[0.0004, 0.0002], [0.0002, 0.0003]],
                                                    size=n)

            prices1 = 100 * np.exp(np.cumsum(returns[:, 0]))
            prices2 = 100 * np.exp(np.cumsum(returns[:, 1]))

            df = pd.DataFrame({
                'Asset1': prices1,
                'Asset2': prices2
            }, index=dates)
            print("✅ Dataset di esempio creato")

    print(f"\nDimensioni del dataset: {df.shape}")
    print(f"Periodo: {df.index.min()} - {df.index.max()}")
    print(f"Colonne: {df.columns.tolist()}")
    print("\nPrime 5 righe:")
    print(df.head())

    # 2. Preparazione dei dati per l'analisi delle copule
    print("\n2️⃣ Preparazione dei dati")
    print("-" * 50)

    # Seleziona due colonne per l'analisi
    if len(df.columns) < 2:
        raise ValueError("Sono necessarie almeno due colonne di dati")

    # Prova a identificare colonne di prezzo
    price_cols = [col for col in df.columns if col in
                  ['Open', 'Close']]#, 'High', 'Low', 'Adj Close', 'Price', 'Asset1', 'Asset2'

    if len(price_cols) >= 2:
        col1, col2 = price_cols[:2]
    else:
        # Usa le prime due colonne numeriche disponibili
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[:2]
        else:
            raise ValueError("Non sufficienti colonne numeriche nel dataset")

    print(f"Utilizzo delle colonne: '{col1}' e '{col2}'")

    # Conversione in valori numerici
    df[col1] = pd.to_numeric(df[col1], errors='coerce')
    df[col2] = pd.to_numeric(df[col2], errors='coerce')

    # Rimozione di valori mancanti
    df = df.dropna(subset=[col1, col2])

    # Calcolo dei rendimenti logaritmici
    returns = pd.DataFrame({
        'r1': np.log(df[col1] / df[col1].shift(1)),
        'r2': np.log(df[col2] / df[col2].shift(1))
    }).dropna()

    print(f"Calcolati {len(returns)} rendimenti logaritmici")

    # Statistiche descrittive dei rendimenti
    print("\nStatistiche descrittive dei rendimenti:")
    print(returns.describe())

    # Visualizzazione della distribuzione dei rendimenti
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(returns['r1'], kde=True, label=col1, color='blue', alpha=0.7)
    plt.title(f'Distribuzione rendimenti {col1}')
    plt.xlabel('Rendimento')
    plt.ylabel('Frequenza')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.histplot(returns['r2'], kde=True, label=col2, color='green', alpha=0.7)
    plt.title(f'Distribuzione rendimenti {col2}')
    plt.xlabel('Rendimento')
    plt.ylabel('Frequenza')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('returns_distribution.png', dpi=300)

    # 3. Calcolo della volatilità rolling
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
    print(f"Calcolata volatilità rolling su finestra di {window} giorni")
    print(f"Volatilità media di {col1}: {vol_returns['vol1'].mean():.2%}")
    print(f"Volatilità media di {col2}: {vol_returns['vol2'].mean():.2%}")

    # # 3. Calcolo della volatilità rolling - OTTIMIZZATO
    # print("\n3️⃣ Calcolo della volatilità rolling")
    # print("-" * 50)
    # print("🕒 Questo potrebbe richiedere del tempo con dataset grandi...")

    # # Per dataset molto grandi, potrebbe essere utile ridurre la dimensione
    # # Verifichiamo se il dataset è molto grande e lo campionamo se necessario
    # if len(returns) > 100000:
    #     print(f"⚙️ Campionamento del dataset da {len(returns)} a ~100000 righe per migliorare le prestazioni")
    #     # Campionamento del dataset per migliorare le prestazioni
    #     sample_ratio = max(1, len(returns) // 100000)
    #     returns_sampled = returns.iloc[::sample_ratio].copy()
    #     print(f"✅ Dataset ridotto a {len(returns_sampled)} righe")
    # else:
    #     returns_sampled = returns.copy()

    # # Definizione della finestra per la volatilità rolling (es. 21 giorni = 1 mese di trading)
    # window = 21
    # vol_window = min(window, len(returns_sampled) // 4)  # Assicura una finestra ragionevole

    # print(f"⚙️ Calcolo volatilità rolling con finestra di {vol_window} giorni...")
    
    # # Calcolo della volatilità rolling - più efficiente usando numba se disponibile
    # try:
    #     # Proviamo a usare numba per accelerare se è installato
    #     import numba
    #     print("✅ Usando numba per accelerare il calcolo")
        
    #     @numba.jit(nopython=True)
    #     def rolling_std(arr, window):
    #         n = len(arr)
    #         result = np.empty(n)
    #         result[:window-1] = np.nan
            
    #         for i in range(window-1, n):
    #             window_slice = arr[i-window+1:i+1]
    #             result[i] = np.std(window_slice)
                
    #         return result
        
    #     # Calcolo con numba
    #     vol1 = rolling_std(returns_sampled['r1'].values, vol_window) * np.sqrt(252)
    #     vol2 = rolling_std(returns_sampled['r2'].values, vol_window) * np.sqrt(252)
        
    #     returns_sampled['vol1'] = vol1
    #     returns_sampled['vol2'] = vol2
    # except ImportError:
    #     # Fallback a pandas se numba non è disponibile
    #     print("⚠️ numba non disponibile, usando pandas (più lento)")
    #     returns_sampled['vol1'] = returns_sampled['r1'].rolling(window=vol_window).std() * np.sqrt(252)
    #     returns_sampled['vol2'] = returns_sampled['r2'].rolling(window=vol_window).std() * np.sqrt(252)

    # # Rimozione delle prime righe con NaN dovuti alla finestra rolling
    # vol_returns = returns_sampled.dropna()
    # print(f"✅ Volatilità calcolata su {len(vol_returns)} punti")

    # print(f"Calcolata volatilità rolling su finestra di {vol_window} giorni")
    # print(f"Volatilità media di {col1}: {vol_returns['vol1'].mean():.2%}")
    # print(f"Volatilità media di {col2}: {vol_returns['vol2'].mean():.2%}")

    # # Visualizzazione della volatilità rolling
    # plt.figure(figsize=(12, 6))
    # plt.plot(vol_returns.index, vol_returns['vol1'], label=f'{col1} Volatility', color='blue')
    # plt.plot(vol_returns.index, vol_returns['vol2'], label=f'{col2} Volatility', color='green')
    # plt.title('Volatilità Rolling Annualizzata')
    # plt.xlabel('Data')
    # plt.ylabel('Volatilità')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.savefig('rolling_volatility.png', dpi=300)

    # 4. Trasformazione in distribuzioni marginali uniformi
    print("\n4️⃣ Trasformazione in distribuzioni uniformi per l'analisi delle copule")
    print("-" * 50)

    # Utilizziamo il ranking empirico per mappare i dati in [0,1]
    u1 = stats.rankdata(vol_returns['vol1']) / (len(vol_returns) + 1)
    u2 = stats.rankdata(vol_returns['vol2']) / (len(vol_returns) + 1)

    # Visualizzazione dei dati uniformati
    plt.figure(figsize=(10, 8))
    plt.scatter(u1, u2, alpha=0.7)
    plt.title('Scatter Plot delle Volatilità Uniformate')
    plt.xlabel(f'{col1} Volatility (Uniform)')
    plt.ylabel(f'{col2} Volatility (Uniform)')
    plt.grid(True, alpha=0.3)
    plt.savefig('uniform_volatility.png', dpi=300)

    # Calcolo delle misure di dipendenza
    print("\nMisure di dipendenza tra le volatilità:")
    pearson_corr = np.corrcoef(vol_returns['vol1'], vol_returns['vol2'])[0, 1]
    kendall_tau, _ = stats.kendalltau(vol_returns['vol1'], vol_returns['vol2'])
    spearman_rho, _ = stats.spearmanr(vol_returns['vol1'], vol_returns['vol2'])

    print(f"Correlazione di Pearson: {pearson_corr:.4f}")
    print(f"Tau di Kendall: {kendall_tau:.4f}")
    print(f"Rho di Spearman: {spearman_rho:.4f}")

    # 5. Implementazione delle funzioni di log-verosimiglianza per diverse copule
    print("\n5️⃣ Implementazione delle funzioni di copula")
    print("-" * 50)


    # 5.1 Copula Gaussiana
    def gaussian_copula_loglik(rho, u, v):
        if abs(rho) >= 1:
            return -np.inf

        norm_u = stats.norm.ppf(u)
        norm_v = stats.norm.ppf(v)

        term1 = -0.5 * np.log(1 - rho ** 2)
        term2 = -0.5 * (rho ** 2 * (norm_u ** 2 + norm_v ** 2) - 2 * rho * norm_u * norm_v) / (1 - rho ** 2)

        return np.sum(term1 + term2)


    # 5.2 Copula t-Student
    def t_copula_loglik(params, u, v):
        rho, nu = params

        if abs(rho) >= 1 or nu <= 2:
            return -np.inf

        t_u = stats.t.ppf(u, nu)
        t_v = stats.t.ppf(v, nu)

        w = (t_u ** 2 + t_v ** 2 - 2 * rho * t_u * t_v) / (1 - rho ** 2)
        term1 = -0.5 * np.log(1 - rho ** 2)
        term2 = -(nu + 2) / 2 * np.log(1 + w / nu)
        term3 = -(stats.t.logpdf(t_u, nu) + stats.t.logpdf(t_v, nu))

        return np.sum(term1 + term2 + term3)


    # 5.3 Copula di Clayton
    def clayton_copula_loglik(theta, u, v):
        if theta <= 0:
            return -np.inf

        term1 = np.log(1 + theta)
        term2 = -(1 + theta) * np.log(u * v)
        term3 = -(2 + 1 / theta) * np.log(u ** (-theta) + v ** (-theta) - 1)

        return np.sum(term1 + term2 + term3)


    # 5.4 Copula di Gumbel
    def gumbel_copula_loglik(theta, u, v):
        if theta < 1:
            return -np.inf

        log_u = -np.log(u)
        log_v = -np.log(v)
        w = (log_u ** theta + log_v ** theta) ** (1 / theta)

        term1 = -w
        term2 = np.log(w * (log_u * log_v) ** (theta - 1))
        term3 = np.log(theta - 1 + w) - np.log(u * v)

        return np.sum(term1 + term2 + term3)


    # 5.5 Copula di Frank
    def frank_copula_loglik(theta, u, v):
        if abs(theta) < 1e-10:
            return -np.inf

        term1 = np.log(abs(theta)) - np.log(1 - np.exp(-abs(theta)))
        term2 = -theta * (u + v)
        term3 = -2 * np.log(1 + np.exp(-theta) * (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1))

        return np.sum(term1 + term2 + term3)


    # 6. Stima dei parametri per ciascuna copula
    print("\n6️⃣ Stima dei parametri delle copule")
    print("-" * 50)


    # Funzioni negative per la minimizzazione
    def neg_gaussian_loglik(rho, u, v):
        return -gaussian_copula_loglik(rho[0], u, v)


    def neg_t_loglik(params, u, v):
        return -t_copula_loglik(params, u, v)


    def neg_clayton_loglik(theta, u, v):
        return -clayton_copula_loglik(theta[0], u, v)


    def neg_gumbel_loglik(theta, u, v):
        return -gumbel_copula_loglik(theta[0], u, v)


    def neg_frank_loglik(theta, u, v):
        return -frank_copula_loglik(theta[0], u, v)


    # 6.1 Stima per la copula Gaussiana
    init_rho = pearson_corr
    bounds_gaussian = [(-0.999, 0.999)]
    result_gaussian = minimize(neg_gaussian_loglik, [init_rho], args=(u1, u2),
                               bounds=bounds_gaussian, method='L-BFGS-B')
    rho_mle = result_gaussian.x[0]
    print(f"Copula Gaussiana - ρ: {rho_mle:.4f}, log-verosimiglianza: {-result_gaussian.fun:.4f}")

    # 6.2 Stima per la copula t-Student
    init_params_t = [pearson_corr, 5]
    bounds_t = [(-0.999, 0.999), (2.001, 30)]
    result_t = minimize(neg_t_loglik, init_params_t, args=(u1, u2),
                        bounds=bounds_t, method='L-BFGS-B')
    rho_t_mle, nu_t_mle = result_t.x
    print(f"Copula t-Student - ρ: {rho_t_mle:.4f}, ν: {nu_t_mle:.4f}, log-verosimiglianza: {-result_t.fun:.4f}")

    # 6.3 Stima per la copula di Clayton
    result_clayton = None
    if kendall_tau > 0:
        init_theta_clayton = (2 * kendall_tau) / (1 - kendall_tau)
        bounds_clayton = [(0.001, 20)]
        result_clayton = minimize(neg_clayton_loglik, [init_theta_clayton], args=(u1, u2),
                                  bounds=bounds_clayton, method='L-BFGS-B')
        theta_clayton_mle = result_clayton.x[0]
        print(f"Copula di Clayton - θ: {theta_clayton_mle:.4f}, log-verosimiglianza: {-result_clayton.fun:.4f}")
    else:
        print("Copula di Clayton non applicabile (τ ≤ 0)")

    # 6.4 Stima per la copula di Gumbel
    result_gumbel = None
    if kendall_tau > 0:
        init_theta_gumbel = 1 / (1 - kendall_tau)
        bounds_gumbel = [(1.001, 20)]
        result_gumbel = minimize(neg_gumbel_loglik, [init_theta_gumbel], args=(u1, u2),
                                 bounds=bounds_gumbel, method='L-BFGS-B')
        theta_gumbel_mle = result_gumbel.x[0]
        print(f"Copula di Gumbel - θ: {theta_gumbel_mle:.4f}, log-verosimiglianza: {-result_gumbel.fun:.4f}")
    else:
        print("Copula di Gumbel non applicabile (τ ≤ 0)")

    # 6.5 Stima per la copula di Frank
    init_theta_frank = 0.5
    bounds_frank = [(-20, 20)]
    result_frank = minimize(neg_frank_loglik, [init_theta_frank], args=(u1, u2),
                            bounds=bounds_frank, method='L-BFGS-B')
    theta_frank_mle = result_frank.x[0]
    print(f"Copula di Frank - θ: {theta_frank_mle:.4f}, log-verosimiglianza: {-result_frank.fun:.4f}")

    # 7. Confronto dei modelli e selezione della copula migliore
    print("\n7️⃣ Confronto dei modelli con AIC e BIC")
    print("-" * 50)

    # Numero di parametri per ciascun modello
    n_params_gaussian = 1
    n_params_t = 2
    n_params_clayton = 1
    n_params_gumbel = 1
    n_params_frank = 1

    # Dimensione del campione
    n_samples = len(u1)

    # Calcolo dell'AIC: -2*log-likelihood + 2*k
    aic_gaussian = 2 * result_gaussian.fun + 2 * n_params_gaussian
    aic_t = 2 * result_t.fun + 2 * n_params_t

    models = ['Gaussiana', 't-Student', 'Clayton', 'Gumbel', 'Frank']
    log_likelihoods = [-result_gaussian.fun, -result_t.fun]
    aics = [aic_gaussian, aic_t]
    bics = [2 * result_gaussian.fun + n_params_gaussian * np.log(n_samples),
            2 * result_t.fun + n_params_t * np.log(n_samples)]
    params = [f"ρ={rho_mle:.4f}", f"ρ={rho_t_mle:.4f}, ν={nu_t_mle:.4f}"]

    if kendall_tau > 0 and result_clayton is not None:
        aic_clayton = 2 * result_clayton.fun + 2 * n_params_clayton
        aics.append(aic_clayton)
        log_likelihoods.append(-result_clayton.fun)
        bics.append(2 * result_clayton.fun + n_params_clayton * np.log(n_samples))
        params.append(f"θ={theta_clayton_mle:.4f}")
    else:
        models.remove('Clayton')

    if kendall_tau > 0 and result_gumbel is not None:
        aic_gumbel = 2 * result_gumbel.fun + 2 * n_params_gumbel
        aics.append(aic_gumbel)
        log_likelihoods.append(-result_gumbel.fun)
        bics.append(2 * result_gumbel.fun + n_params_gumbel * np.log(n_samples))
        params.append(f"θ={theta_gumbel_mle:.4f}")
    else:
        models.remove('Gumbel')

    aic_frank = 2 * result_frank.fun + 2 * n_params_frank
    aics.append(aic_frank)
    log_likelihoods.append(-result_frank.fun)
    bics.append(2 * result_frank.fun + n_params_frank * np.log(n_samples))
    params.append(f"θ={theta_frank_mle:.4f}")

    # Creazione di un DataFrame per il confronto
    results_df = pd.DataFrame({
        'Modello': models,
        'Parametri': params,
        'Log-Verosimiglianza': log_likelihoods,
        'AIC': aics,
        'BIC': bics
    })

    # Ordinamento per AIC crescente (il più basso è il migliore)
    results_df = results_df.sort_values('AIC')
    print(results_df)

    # Salva i risultati
    results_df.to_csv('copula_selection_results.csv', index=False)
    print("\n✅ Risultati della selezione salvati in 'copula_selection_results.csv'")

    # Identifica il miglior modello
    best_model = results_df.iloc[0]
    print(f"\nIl modello migliore è la copula {best_model['Modello']} con {best_model['Parametri']}")
    print(f"AIC: {best_model['AIC']:.4f}, BIC: {best_model['BIC']:.4f}")

    # 8. Simulazione dalla copula selezionata
    print("\n8️⃣ Simulazione dalla copula selezionata")
    print("-" * 50)

    # Numero di simulazioni
    n_sim = 1000
    selected_model = best_model['Modello']

    # Simulazione in base al modello selezionato
    if selected_model == 'Gaussiana':
        rho = rho_mle
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        z = np.random.multivariate_normal(mean, cov, n_sim)
        simulated_uniform = stats.norm.cdf(z)

    elif selected_model == 't-Student':
        rho, nu = rho_t_mle, nu_t_mle
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        z = np.random.multivariate_normal(mean, cov, n_sim)

        # Genera gradi di libertà per t
        w = np.random.chisquare(nu, n_sim) / nu
        z = z / np.sqrt(w)[:, np.newaxis]

        simulated_uniform = stats.t.cdf(z, nu)

    elif selected_model == 'Clayton':
        theta = theta_clayton_mle
        # Genera dalla copula di Clayton
        v1 = np.random.uniform(0, 1, n_sim)
        v2 = np.random.uniform(0, 1, n_sim)

        u1_sim = v1
        u2_sim = (1 + (v1 ** (-theta) - 1) / (v2 ** (-1 / theta) - 1)) ** (-1 / theta)

        simulated_uniform = np.column_stack((u1_sim, u2_sim))

    elif selected_model == 'Gumbel':
        theta = theta_gumbel_mle
        # Genera dalla copula di Gumbel (approssimazione)
        v1 = np.random.uniform(0, 1, n_sim)
        v2 = np.random.uniform(0, 1, n_sim)

        # Questa è un'approssimazione
        from scipy.stats import genextreme

        gamma = -1 / theta
        z = genextreme.rvs(gamma, size=n_sim)
        u1_sim = np.exp(-np.exp(-z))
        u2_sim = np.exp(-np.exp(-z - np.log(v2)))

        simulated_uniform = np.column_stack((u1_sim, u2_sim))

    elif selected_model == 'Frank':
        theta = theta_frank_mle
        # Genera dalla copula di Frank
        v1 = np.random.uniform(0, 1, n_sim)
        v2 = np.random.uniform(0, 1, n_sim)

        u1_sim = v1
        if abs(theta) < 1e-10:
            u2_sim = v2  # Indipendenza
        else:
            u2_sim = -np.log(1 + (v2 * (1 - np.exp(-theta))) /
                             (np.exp(-theta * v1) * (np.exp(-theta) - 1) + (1 - np.exp(-theta)))) / theta

        simulated_uniform = np.column_stack((u1_sim, u2_sim))

    # 9. Visualizzazione dei risultati: originali vs simulati
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(u1, u2, alpha=0.7)
    plt.title('Volatilità Originali Uniformate')
    plt.xlabel(f'{col1} Volatility (Unif)')
    plt.ylabel(f'{col2} Volatility (Unif)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(simulated_uniform[:, 0], simulated_uniform[:, 1], alpha=0.7, color='red')
    plt.title(f'Copula {selected_model} Simulata\n{best_model["Parametri"]}')
    plt.xlabel('U1')
    plt.ylabel('U2')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulated_volatility.png', dpi=300)
    print("✅ Grafico della simulazione salvato come 'simulated_volatility.png'")

    # 10. Implicazioni per l'ottimizzazione di portafoglio
    print("\n9️⃣ Implicazioni per l'ottimizzazione di portafoglio")
    print("-" * 50)

    # Calcola media e varianza dei rendimenti originali
    mean_returns = returns[['r1', 'r2']].mean()
    cov_returns = returns[['r1', 'r2']].cov()

    print("Statistiche dei rendimenti originali:")
    print(f"Media giornaliera {col1}: {mean_returns['r1'] * 100:.4f}%")
    print(f"Media giornaliera {col2}: {mean_returns['r2'] * 100:.4f}%")
    print(f"Volatilità annualizzata {col1}: {np.sqrt(252) * np.sqrt(cov_returns.iloc[0, 0]) * 100:.2f}%")
    print(f"Volatilità annualizzata {col2}: {np.sqrt(252) * np.sqrt(cov_returns.iloc[1, 1]) * 100:.2f}%")
    print(f"Correlazione: {cov_returns.iloc[0, 1] / np.sqrt(cov_returns.iloc[0, 0] * cov_returns.iloc[1, 1]):.4f}")

    # Calcolo di portafogli ottimali con diverse allocazioni
    weights = np.linspace(0, 1, 101)  # Da 0% a 100% in Asset1

    # Calcolo del rendimento e rischio attesi per ciascuna allocazione
    portfolio_returns = []
    portfolio_risks = []
    sharpe_ratios = []
    risk_free_rate = 0.02 / 252  # Tasso risk-free giornaliero (2% annuo)

    for w1 in weights:
        w2 = 1 - w1
        w = np.array([w1, w2])

        # Rendimento atteso del portafoglio
        port_return = np.dot(w, mean_returns)

        # Rischio del portafoglio
        port_risk = np.sqrt(np.dot(w.T, np.dot(cov_returns, w)))

        # Sharpe ratio
        sharpe = (port_return - risk_free_rate) / port_risk

        portfolio_returns.append(port_return * 252)  # Annualizzato
        portfolio_risks.append(port_risk * np.sqrt(252))  # Annualizzato
        sharpe_ratios.append(sharpe * np.sqrt(252))  # Annualizzato

    # Trova il portafoglio con il massimo Sharpe ratio
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_return = portfolio_returns[max_sharpe_idx]
    max_sharpe_risk = portfolio_risks[max_sharpe_idx]
    max_sharpe_weight = weights[max_sharpe_idx]

    print(f"\nPortafoglio con massimo Sharpe Ratio:")
    print(f"Allocazione in {col1}: {max_sharpe_weight * 100:.2f}%")
    print(f"Allocazione in {col2}: {(1 - max_sharpe_weight) * 100:.2f}%")
    print(f"Rendimento atteso annualizzato: {max_sharpe_return * 100:.2f}%")
    print(f"Rischio annualizzato: {max_sharpe_risk * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.4f}")

    # Trova il portafoglio con minima varianza
    min_risk_idx = np.argmin(portfolio_risks)
    min_risk_return = portfolio_returns[min_risk_idx]
    min_risk_vol = portfolio_risks[min_risk_idx]
    min_risk_weight = weights[min_risk_idx]

    print(f"\nPortafoglio a varianza minima:")
    print(f"Allocazione in {col1}: {min_risk_weight * 100:.2f}%")
    print(f"Allocazione in {col2}: {(1 - min_risk_weight) * 100:.2f}%")
    print(f"Rendimento atteso annualizzato: {min_risk_return * 100:.2f}%")
    print(f"Rischio annualizzato: {min_risk_vol * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratios[min_risk_idx]:.4f}")

    # Visualizzazione della frontiera efficiente
    plt.figure(figsize=(12, 8))

    # Frontiera efficiente
    plt.scatter(portfolio_risks, portfolio_returns,
                c=sharpe_ratios, cmap='viridis',
                alpha=0.7, marker='o')

    # Portafoglio con massimo Sharpe ratio
    plt.scatter(max_sharpe_risk, max_sharpe_return,
                marker='*', color='red', s=300,
                label=f'Max Sharpe: {max_sharpe_weight * 100:.0f}% in {col1}')

    # Portafoglio a minima varianza
    plt.scatter(min_risk_vol, min_risk_return,
                marker='*', color='gold', s=300,
                label=f'Min Volatilità: {min_risk_weight * 100:.0f}% in {col1}')

    # Asset individuali
    asset1_risk = portfolio_risks[100]  # 100% in Asset1
    asset1_return = portfolio_returns[100]
    asset2_risk = portfolio_risks[0]  # 100% in Asset2
    asset2_return = portfolio_returns[0]

    plt.scatter(asset1_risk, asset1_return,
                marker='D', color='blue', s=150,
                label=f'{col1}')
    plt.scatter(asset2_risk, asset2_return,
                marker='D', color='green', s=150,
                label=f'{col2}')

    plt.colorbar(label='Sharpe Ratio')
    plt.title('Frontiera di Efficienza del Portafoglio')
    plt.xlabel('Rischio Annualizzato (Volatilità)')
    plt.ylabel('Rendimento Atteso Annualizzato')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('efficient_frontier.png', dpi=300)
    print("✅ Grafico della frontiera efficiente salvato come 'efficient_frontier.png'")

    # 11. Analisi di Stress Testing usando la copula selezionata
    print("\n🔟 Stress Testing con la copula selezionata")
    print("-" * 50)

    # Simulazione di scenari estremi basati sulla copula selezionata
    n_scenarios = 5000

    # Genera simulazioni dalla copula selezionata
    if selected_model == 'Gaussiana':
        rho = rho_mle
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        z = np.random.multivariate_normal(mean, cov, n_scenarios)
        simulated_unif = stats.norm.cdf(z)

    elif selected_model == 't-Student':
        rho, nu = rho_t_mle, nu_t_mle
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        z = np.random.multivariate_normal(mean, cov, n_scenarios)

        # Genera gradi di libertà per t
        w = np.random.chisquare(nu, n_scenarios) / nu
        z = z / np.sqrt(w)[:, np.newaxis]

        simulated_unif = stats.t.cdf(z, nu)

    elif selected_model == 'Clayton':
        theta = theta_clayton_mle
        # Genera dalla copula di Clayton
        v1 = np.random.uniform(0, 1, n_scenarios)
        v2 = np.random.uniform(0, 1, n_scenarios)

        u1_sim = v1
        u2_sim = (1 + (v1 ** (-theta) - 1) / (v2 ** (-1 / theta) - 1)) ** (-1 / theta)

        simulated_unif = np.column_stack((u1_sim, u2_sim))

    elif selected_model == 'Gumbel':
        theta = theta_gumbel_mle
        # Genera dalla copula di Gumbel (approssimazione)
        v1 = np.random.uniform(0, 1, n_scenarios)
        v2 = np.random.uniform(0, 1, n_scenarios)

        # Questa è un'approssimazione
        from scipy.stats import genextreme

        gamma = -1 / theta
        z = genextreme.rvs(gamma, size=n_scenarios)
        u1_sim = np.exp(-np.exp(-z))
        u2_sim = np.exp(-np.exp(-z - np.log(v2)))

        simulated_unif = np.column_stack((u1_sim, u2_sim))

    else:  # Frank
        theta = theta_frank_mle
        # Genera dalla copula di Frank
        v1 = np.random.uniform(0, 1, n_scenarios)
        v2 = np.random.uniform(0, 1, n_scenarios)

        u1_sim = v1
        if abs(theta) < 1e-10:
            u2_sim = v2  # Indipendenza
        else:
            u2_sim = -np.log(1 + (v2 * (1 - np.exp(-theta))) /
                             (np.exp(-theta * v1) * (np.exp(-theta) - 1) + (1 - np.exp(-theta)))) / theta

        simulated_unif = np.column_stack((u1_sim, u2_sim))

    # Trasforma le simulazioni uniformi in simulazioni di volatilità
    # Usiamo l'inversa della CDF empirica (i quantili)
    from scipy.interpolate import interp1d


    # Funzione per trasformare valori uniformi in valori originali usando la distribuzione empirica
    def inverse_ecdf(uniform_values, original_data):
        """Trasforma valori uniformi in valori dalla distribuzione empirica."""
        # Ordina i dati originali
        sorted_data = np.sort(original_data)

        # Crea una funzione di interpolazione dai quantili uniformi ai valori ordinati
        quantiles = np.linspace(0, 1, len(sorted_data))
        inv_cdf = interp1d(quantiles, sorted_data, bounds_error=False, fill_value=(sorted_data[0], sorted_data[-1]))

        # Applica la funzione inversa
        return inv_cdf(uniform_values)


    # Trasforma le simulazioni uniform in valori di volatilità
    vol1_sim = inverse_ecdf(simulated_unif[:, 0], vol_returns['vol1'])
    vol2_sim = inverse_ecdf(simulated_unif[:, 1], vol_returns['vol2'])

    # Calcolo di scenari di stress per il portafoglio
    # Assumiamo che il portafoglio ottimale sia quello con massimo Sharpe Ratio
    opt_weight1 = max_sharpe_weight
    opt_weight2 = 1 - opt_weight1

    # Calcoliamo la volatilità del portafoglio in ogni scenario simulato
    # Nota: questa è una semplificazione, in realtà dovremmo considerare anche la correlazione
    # che varia nel tempo, ma per semplicità assumiamo la correlazione fissa stimata dal modello

    port_vol_sim = np.sqrt(opt_weight1 ** 2 * vol1_sim ** 2 +
                           opt_weight2 ** 2 * vol2_sim ** 2 +
                           2 * opt_weight1 * opt_weight2 * rho_mle * vol1_sim * vol2_sim)

    # Calcolo delle statistiche di stress
    vol_threshold = np.percentile(port_vol_sim, 95)  # 95° percentile
    stressed_scenarios = port_vol_sim >= vol_threshold

    print(f"Analisi di stress basata su {n_scenarios} simulazioni:")
    print(f"Volatilità media del portafoglio ottimale: {np.mean(port_vol_sim):.2%}")
    print(f"Volatilità massima simulata: {np.max(port_vol_sim):.2%}")
    print(f"Volatilità in condizioni di stress (95° percentile): {vol_threshold:.2%}")
    print(f"Numero di scenari di stress: {np.sum(stressed_scenarios)}")

    # Visualizzazione degli scenari di stress
    plt.figure(figsize=(12, 8))

    # Scatter plot delle volatilità simulate
    plt.scatter(vol1_sim, vol2_sim, alpha=0.3, s=10,
                c=np.where(stressed_scenarios, 'red', 'blue'),
                label='Scenario normale' if np.sum(~stressed_scenarios) > 0 else None)

    # Evidenzia gli scenari di stress
    if np.sum(stressed_scenarios) > 0:
        plt.scatter(vol1_sim[stressed_scenarios], vol2_sim[stressed_scenarios],
                    alpha=0.7, s=20, c='red', label='Scenario di stress')

    plt.title('Scenari di Stress Testing')
    plt.xlabel(f'Volatilità {col1}')
    plt.ylabel(f'Volatilità {col2}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('stress_testing.png', dpi=300)
    print("✅ Grafico degli scenari di stress salvato come 'stress_testing.png'")

    # 12. Previsione della volatilità futura
    print("\n1️⃣1️⃣ Previsione della volatilità futura")
    print("-" * 50)

    # Creiamo un modello GARCH per prevedere la volatilità futura
    try:
        import arch

        # Funzione per stimare e prevedere con GARCH
        def fit_garch(returns, p=1, q=1, horizon=22):
            model = arch.arch_model(returns, p=p, q=q, mean='Constant', vol='GARCH', dist='normal')
            results = model.fit(disp='off')
            forecasts = results.forecast(horizon=horizon)
            return results, forecasts

        # Adatta modelli GARCH a ciascuna serie di rendimenti
        print("Stima modelli GARCH(1,1) per prevedere la volatilità futura...")

        model1, forecast1 = fit_garch(returns['r1'])
        model2, forecast2 = fit_garch(returns['r2'])

        # Estrai le previsioni di volatilità
        vol_forecast1 = np.sqrt(forecast1.variance.iloc[-1].values)
        vol_forecast2 = np.sqrt(forecast2.variance.iloc[-1].values)

        # Calcola la volatilità prevista annualizzata
        annual_vol_forecast1 = vol_forecast1 * np.sqrt(252)
        annual_vol_forecast2 = vol_forecast2 * np.sqrt(252)

        # *** FIX: Correzione per evitare errore di timezone ***
        # Genera date future per la visualizzazione usando indici numerici invece di date
        # per evitare problemi con le timezones
        
        # Visualizzazione delle previsioni di volatilità
        plt.figure(figsize=(12, 6))

        # Visualizza ultimi 60 valori storici
        plt.plot(range(-60, 0), vol_returns['vol1'][-60:].values,
                 label=f'{col1} Volatilità storica', color='blue', alpha=0.7)
        plt.plot(range(-60, 0), vol_returns['vol2'][-60:].values,
                 label=f'{col2} Volatilità storica', color='green', alpha=0.7)

        # Visualizza previsioni con indici numerici
        x_forecast = range(0, len(annual_vol_forecast1))
        plt.plot(x_forecast, annual_vol_forecast1, '--',
                 label=f'{col1} Previsione', color='blue', linewidth=2)
        plt.plot(x_forecast, annual_vol_forecast2, '--',
                 label=f'{col2} Previsione', color='green', linewidth=2)

        plt.title('Previsione della Volatilità (GARCH)')
        plt.xlabel('Giorni (0 = Oggi, Valori Negativi = Dati Storici)')
        plt.ylabel('Volatilità Annualizzata')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('volatility_forecast.png', dpi=300)
        print("✅ Grafico delle previsioni di volatilità salvato come 'volatility_forecast.png'")

        # Calcoliamo la volatilità prevista del portafoglio ottimale
        # usando la copula stimata per modellare la dipendenza

        # Prima, convertiamo le previsioni di volatilità in distribuzioni marginali uniformi
        u1_forecast = stats.rankdata(annual_vol_forecast1) / (len(annual_vol_forecast1) + 1)
        u2_forecast = stats.rankdata(annual_vol_forecast2) / (len(annual_vol_forecast2) + 1)

        # Calcoliamo la correlazione prevista usando la copula selezionata
        if selected_model == 'Gaussiana':
            # Per la copula Gaussiana, la correlazione è direttamente il parametro rho
            forecast_corr = rho_mle
        elif selected_model == 't-Student':
            # Per la t-Student, la correlazione è il parametro rho
            forecast_corr = rho_t_mle
        else:
            # Per altre copule, possiamo calcolare il tau di Kendall dalle simulazioni
            # e poi convertirlo in correlazione usando la relazione di Pearson
            kendall_forecast, _ = stats.kendalltau(u1_forecast, u2_forecast)
            forecast_corr = np.sin(np.pi * kendall_forecast / 2)  # Approssimazione

        # Calcolo della volatilità prevista del portafoglio ottimale
        port_vol_forecast = []

        for i in range(len(annual_vol_forecast1)):
            vol1 = annual_vol_forecast1[i]
            vol2 = annual_vol_forecast2[i]

            port_vol = np.sqrt(opt_weight1 ** 2 * vol1 ** 2 +
                               opt_weight2 ** 2 * vol2 ** 2 +
                               2 * opt_weight1 * opt_weight2 * forecast_corr * vol1 * vol2)

            port_vol_forecast.append(port_vol)

        # Visualizzazione della volatilità prevista del portafoglio
        plt.figure(figsize=(12, 6))

        plt.plot(x_forecast, port_vol_forecast, '-',
                 label='Volatilità Portafoglio', color='purple', linewidth=2)

        # Aggiungi intervallo di confidenza basato sulle simulazioni
        port_vol_percentiles = np.percentile(port_vol_sim, [5, 95])
        plt.axhline(y=port_vol_percentiles[0], color='orange', linestyle=':', label='5° Percentile')
        plt.axhline(y=port_vol_percentiles[1], color='red', linestyle=':', label='95° Percentile')

        plt.title('Previsione della Volatilità del Portafoglio Ottimale')
        plt.xlabel('Giorni nel Futuro')
        plt.ylabel('Volatilità Annualizzata')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('portfolio_volatility_forecast.png', dpi=300)
        print("✅ Grafico della volatilità prevista del portafoglio salvato come 'portfolio_volatility_forecast.png'")

        print("\nPrevisione della volatilità per i prossimi 22 giorni:")
        print(f"Media {col1}: {np.mean(annual_vol_forecast1):.2%}")
        print(f"Media {col2}: {np.mean(annual_vol_forecast2):.2%}")
        print(f"Media Portafoglio Ottimale: {np.mean(port_vol_forecast):.2%}")
        print(f"Intervallo di confidenza (5%-95%): [{port_vol_percentiles[0]:.2%}, {port_vol_percentiles[1]:.2%}]")

    except ImportError:
        print("⚠️ Modulo 'arch' non disponibile per la modellazione GARCH. Installa con 'pip install arch'.")

    # 13. Conclusioni e Raccomandazioni
    print("\n1️⃣2️⃣ Conclusioni e Raccomandazioni")
    print("-" * 50)

    print(f"✅ La copula {selected_model} è stata identificata come il modello migliore")
    print(f"✅ Parametri stimati: {best_model['Parametri']}")

    # Implicazioni per la gestione del portafoglio
    print("\nImplicazioni per la gestione del portafoglio:")

    # Analisi della dipendenza delle code
    if selected_model == 't-Student':
        print("- La copula t-Student indica una significativa dipendenza delle code tra le volatilità")
        print(f"- Con {nu_t_mle:.2f} gradi di libertà, c'è un rischio elevato di movimenti estremi simultanei")
        print("- Si consiglia di implementare strategie di hedging per mitigare il rischio delle code")
    elif selected_model == 'Clayton':
        print("- La copula di Clayton indica forte dipendenza nella coda inferiore")
        print("- C'è un rischio elevato che entrambi gli asset subiscano contemporaneamente cali di volatilità")
        print("- Potrebbe essere necessaria una strategia di copertura nei mercati in calo")
    elif selected_model == 'Gumbel':
        print("- La copula di Gumbel indica forte dipendenza nella coda superiore")
        print("- C'è un rischio elevato che entrambi gli asset subiscano contemporaneamente picchi di volatilità")
        print("- Si consiglia cautela durante i periodi di alta volatilità del mercato")
    elif selected_model == 'Frank':
        print("- La copula di Frank indica dipendenza simmetrica senza forte dipendenza delle code")
        print("- La diversificazione tra i due asset offre buoni benefici in condizioni normali")
    else:  # Gaussiana
        print("- La copula Gaussiana non cattura la dipendenza delle code")
        print("- I benefici della diversificazione potrebbero essere sovrastimati durante stress di mercato")
        print("- Si consiglia cautela nel fare affidamento esclusivamente su questa modellazione")

    # Allocation ottimale
    print(f"\nAllocazione ottimale del portafoglio:")
    print(f"- {col1}: {max_sharpe_weight * 100:.2f}%")
    print(f"- {col2}: {(1 - max_sharpe_weight) * 100:.2f}%")
    print(f"- Rendimento atteso annualizzato: {max_sharpe_return * 100:.2f}%")
    print(f"- Rischio annualizzato: {max_sharpe_risk * 100:.2f}%")
    print(f"- Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.4f}")

    # Raccomandazioni finali
    print("\nRaccomandazioni finali:")
    print("1. Monitorare costantemente la struttura di dipendenza tra gli asset")
    print("2. Ricalibrare il modello di copula durante periodi di alta volatilità")
    print("3. Implementare stress test regolari basati sulla copula selezionata")
    print("4. Considerare strategie di copertura per mitigare i rischi delle code")

    print("\n✅ Analisi completata con successo!")

except Exception as e:
    import traceback

    print(f"❌ Si è verificato un errore durante l'esecuzione:")
    print(traceback.format_exc())