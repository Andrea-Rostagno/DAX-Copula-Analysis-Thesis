import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def simulate_long_portfolio():
    """
    Simulazione di un portafoglio long utilizzando dati DAX e copule stimate
    """
    print("\n📊 SIMULAZIONE PORTAFOGLIO LONG CON COPULE")
    print("=" * 70)
    
    # 1. Impostazione dei parametri di simulazione
    print("\n1️⃣ Impostazione parametri di simulazione")
    print("-" * 50)
    
    # Rendimenti attesi annualizzati positivi, basati su stime a lungo termine
    # Per il DAX, un rendimento medio annuo del 6-8% è storicamente plausibile
    expected_return_open = 0.07  # 7% annuo
    expected_return_close = 0.075  # 7.5% annuo
    
    # Volatilità dai dati originali (dall'output precedente)
    volatility_open = 0.0079  # 0.79% annualizzato
    volatility_close = 0.0081  # 0.81% annualizzato
    
    # Correlazione, possiamo usare quella stimata dalla copula
    # Supponiamo che la copula Gaussiana abbia dato rho = 0.95
    correlation = 0.95
    
    # Tasso risk-free
    risk_free_rate = 0.02  # 2% annuo
    
    # Parametri per la simulazione Monte Carlo
    n_simulations = 10000
    time_horizon = 252  # Un anno di trading
    
    print(f"Rendimento atteso Open: {expected_return_open:.2%}")
    print(f"Rendimento atteso Close: {expected_return_close:.2%}")
    print(f"Volatilità Open: {volatility_open:.4f}")
    print(f"Volatilità Close: {volatility_close:.4f}")
    print(f"Correlazione: {correlation:.4f}")
    print(f"Tasso risk-free: {risk_free_rate:.2%}")
    print(f"Numero simulazioni: {n_simulations}")
    print(f"Orizzonte temporale: {time_horizon} giorni")
    
    # 2. Ottimizzazione del portafoglio con i nuovi parametri
    print("\n2️⃣ Ottimizzazione del portafoglio")
    print("-" * 50)
    
    # Creiamo la matrice di covarianza
    cov_matrix = np.array([
        [volatility_open**2, correlation * volatility_open * volatility_close],
        [correlation * volatility_open * volatility_close, volatility_close**2]
    ])
    
    # Vettore dei rendimenti attesi
    expected_returns = np.array([expected_return_open, expected_return_close])
    
    # Funzione per calcolare il rischio del portafoglio
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Funzione per calcolare il rendimento del portafoglio
    def portfolio_return(weights, expected_returns):
        return np.sum(weights * expected_returns)
    
    # Funzione per calcolare lo Sharpe Ratio negativo (da minimizzare)
    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        p_ret = portfolio_return(weights, expected_returns)
        p_vol = portfolio_volatility(weights, cov_matrix)
        return -(p_ret - risk_free_rate) / p_vol
    
    # Vincoli: la somma dei pesi deve essere 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Limiti: ogni peso deve essere tra 0 e 1
    bounds = tuple((0, 1) for _ in range(2))
    
    # Ottimizzazione per massimizzare lo Sharpe Ratio
    initial_weights = [0.5, 0.5]
    optimal_sharpe = minimize(neg_sharpe_ratio, initial_weights, 
                             args=(expected_returns, cov_matrix, risk_free_rate),
                             method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights_sharpe = optimal_sharpe['x']
    optimal_return_sharpe = portfolio_return(optimal_weights_sharpe, expected_returns)
    optimal_volatility_sharpe = portfolio_volatility(optimal_weights_sharpe, cov_matrix)
    optimal_sharpe_ratio = (optimal_return_sharpe - risk_free_rate) / optimal_volatility_sharpe
    
    print(f"Portafoglio con massimo Sharpe Ratio:")
    print(f"Allocazione in Open: {optimal_weights_sharpe[0]*100:.2f}%")
    print(f"Allocazione in Close: {optimal_weights_sharpe[1]*100:.2f}%")
    print(f"Rendimento atteso annualizzato: {optimal_return_sharpe*100:.2f}%")
    print(f"Rischio annualizzato: {optimal_volatility_sharpe*100:.2f}%")
    print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")
    
    # Ottimizzazione per il portafoglio a minima varianza
    def min_variance(weights, cov_matrix):
        return portfolio_volatility(weights, cov_matrix)
    
    optimal_variance = minimize(min_variance, initial_weights, 
                               args=(cov_matrix,),
                               method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights_var = optimal_variance['x']
    optimal_return_var = portfolio_return(optimal_weights_var, expected_returns)
    optimal_volatility_var = portfolio_volatility(optimal_weights_var, cov_matrix)
    optimal_sharpe_ratio_var = (optimal_return_var - risk_free_rate) / optimal_volatility_var
    
    print(f"\nPortafoglio a varianza minima:")
    print(f"Allocazione in Open: {optimal_weights_var[0]*100:.2f}%")
    print(f"Allocazione in Close: {optimal_weights_var[1]*100:.2f}%")
    print(f"Rendimento atteso annualizzato: {optimal_return_var*100:.2f}%")
    print(f"Rischio annualizzato: {optimal_volatility_var*100:.2f}%")
    print(f"Sharpe Ratio: {optimal_sharpe_ratio_var:.4f}")
    
    # 3. Simulazione Monte Carlo con la copula stimata
    print("\n3️⃣ Simulazione Monte Carlo con copula")
    print("-" * 50)
    
    # Useremo la copula Gaussiana con il parametro rho stimato
    rho = correlation
    
    # Genera simulazioni dalla copula Gaussiana
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    z = np.random.multivariate_normal(mean, cov, n_simulations)
    simulated_uniform = stats.norm.cdf(z)
    
    # Trasforma le simulazioni uniformi in rendimenti
    # Assumiamo una distribuzione normale per semplicità
    sim_returns_open = stats.norm.ppf(simulated_uniform[:, 0], 
                                     expected_return_open/time_horizon, 
                                     volatility_open/np.sqrt(time_horizon))
    sim_returns_close = stats.norm.ppf(simulated_uniform[:, 1], 
                                      expected_return_close/time_horizon, 
                                      volatility_close/np.sqrt(time_horizon))
    
    # Crea traiettorie di prezzi per entrambi gli asset
    initial_price = 100
    price_paths_open = np.zeros((n_simulations, time_horizon))
    price_paths_close = np.zeros((n_simulations, time_horizon))
    
    for i in range(n_simulations):
        # Inizializza con il prezzo iniziale
        price_paths_open[i, 0] = initial_price
        price_paths_close[i, 0] = initial_price
        
        # Simula la traiettoria dei prezzi
        for t in range(1, time_horizon):
            # Genera nuovi rendimenti per ogni passo temporale
            new_z = np.random.multivariate_normal(mean, cov, 1)
            new_uniform = stats.norm.cdf(new_z)
            
            new_return_open = stats.norm.ppf(new_uniform[0, 0], 
                                           expected_return_open/time_horizon, 
                                           volatility_open/np.sqrt(time_horizon))
            new_return_close = stats.norm.ppf(new_uniform[0, 1], 
                                            expected_return_close/time_horizon, 
                                            volatility_close/np.sqrt(time_horizon))
            
            # Aggiorna i prezzi
            price_paths_open[i, t] = price_paths_open[i, t-1] * (1 + new_return_open)
            price_paths_close[i, t] = price_paths_close[i, t-1] * (1 + new_return_close)
    
    # Simula il portafoglio ottimale
    portfolio_paths = optimal_weights_sharpe[0] * price_paths_open + optimal_weights_sharpe[1] * price_paths_close
    
    # Calcola statistiche delle simulazioni
    final_portfolio_values = portfolio_paths[:, -1]
    expected_final_value = np.mean(final_portfolio_values)
    portfolio_var = np.percentile(final_portfolio_values, 5)  # 5% VaR
    portfolio_cvar = np.mean(final_portfolio_values[final_portfolio_values <= portfolio_var])  # CVaR
    
    print(f"Statistiche del portafoglio simulato (Massimo Sharpe):")
    print(f"Valore iniziale: {initial_price:.2f}")
    print(f"Valore finale atteso: {expected_final_value:.2f} (rendimento: {(expected_final_value/initial_price - 1)*100:.2f}%)")
    print(f"Value at Risk (5%): {portfolio_var:.2f} (perdita massima: {(portfolio_var/initial_price - 1)*100:.2f}%)")
    print(f"Conditional VaR (5%): {portfolio_cvar:.2f}")
    
    # 4. Visualizzazione dei risultati
    print("\n4️⃣ Visualizzazione dei risultati")
    print("-" * 50)
    
    # Visualizza la frontiera efficiente
    plt.figure(figsize=(12, 8))
    
    # Calcola la frontiera efficiente
    weights_range = np.linspace(0, 1, 100)
    portfolio_returns = []
    portfolio_volatilities = []
    
    for w1 in weights_range:
        w2 = 1 - w1
        weights = np.array([w1, w2])
        port_ret = portfolio_return(weights, expected_returns)
        port_vol = portfolio_volatility(weights, cov_matrix)
        portfolio_returns.append(port_ret)
        portfolio_volatilities.append(port_vol)
    
    # Calcola Sharpe Ratio per ogni portafoglio
    sharpe_ratios = [(r - risk_free_rate) / v for r, v in zip(portfolio_returns, portfolio_volatilities)]
    
    # Crea il grafico della frontiera efficiente
    plt.scatter(portfolio_volatilities, portfolio_returns, c=sharpe_ratios, cmap='viridis', marker='o', s=10)
    
    # Aggiungi il portafoglio con massimo Sharpe Ratio
    plt.scatter(optimal_volatility_sharpe, optimal_return_sharpe, c='red', marker='*', s=300, 
               label=f'Max Sharpe: {optimal_weights_sharpe[0]*100:.0f}% in Open')
    
    # Aggiungi il portafoglio a minima varianza
    plt.scatter(optimal_volatility_var, optimal_return_var, c='yellow', marker='*', s=300, 
               label=f'Min Var: {optimal_weights_var[0]*100:.0f}% in Open')
    
    # Asset individuali
    plt.scatter(volatility_open, expected_return_open, c='blue', marker='D', s=100, label='Open')
    plt.scatter(volatility_close, expected_return_close, c='green', marker='D', s=100, label='Close')
    
    # Aggiungi la linea del Capital Market Line
    cml_x = np.linspace(0, max(portfolio_volatilities)*1.2, 100)
    cml_y = risk_free_rate + optimal_sharpe_ratio * cml_x
    plt.plot(cml_x, cml_y, 'k--', label='Capital Market Line')
    
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatilità annualizzata')
    plt.ylabel('Rendimento atteso annualizzato')
    plt.title('Frontiera Efficiente - Portafoglio Long')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('long_portfolio_frontier.png', dpi=300)
    
    # Visualizza alcune traiettorie del portafoglio
    plt.figure(figsize=(12, 6))
    
    # Seleziona alcune traiettorie casuali
    n_paths_to_show = 100
    indices = np.random.choice(n_simulations, n_paths_to_show, replace=False)
    
    for idx in indices:
        plt.plot(portfolio_paths[idx], 'b-', alpha=0.1)
    
    # Aggiungi la traiettoria media
    mean_path = np.mean(portfolio_paths, axis=0)
    plt.plot(mean_path, 'r-', linewidth=2, label='Traiettoria media')
    
    # Aggiungi intervalli di confidenza
    upper_95 = np.percentile(portfolio_paths, 95, axis=0)
    lower_5 = np.percentile(portfolio_paths, 5, axis=0)
    
    plt.fill_between(range(time_horizon), lower_5, upper_95, color='gray', alpha=0.3, label='Intervallo 90%')
    
    plt.xlabel('Giorni di trading')
    plt.ylabel('Valore del portafoglio')
    plt.title('Simulazione Monte Carlo - Portafoglio Ottimale')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('long_portfolio_simulation.png', dpi=300)
    
    # Visualizza la distribuzione dei valori finali
    plt.figure(figsize=(12, 6))
    
    sns.histplot(final_portfolio_values, kde=True, bins=50)
    plt.axvline(x=initial_price, color='r', linestyle='--', label='Valore iniziale')
    plt.axvline(x=portfolio_var, color='orange', linestyle='--', label=f'VaR 5%: {portfolio_var:.2f}')
    plt.axvline(x=expected_final_value, color='green', linestyle='-', label=f'Valore atteso: {expected_final_value:.2f}')
    
    plt.xlabel('Valore finale del portafoglio')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione dei Valori Finali del Portafoglio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('long_portfolio_distribution.png', dpi=300)
    
    print("✅ Grafici salvati come:")
    print("  - long_portfolio_frontier.png")
    print("  - long_portfolio_simulation.png")
    print("  - long_portfolio_distribution.png")
    
    return {
        'optimal_weights_sharpe': optimal_weights_sharpe,
        'optimal_return_sharpe': optimal_return_sharpe,
        'optimal_volatility_sharpe': optimal_volatility_sharpe,
        'optimal_sharpe_ratio': optimal_sharpe_ratio,
        'optimal_weights_var': optimal_weights_var,
        'optimal_return_var': optimal_return_var,
        'optimal_volatility_var': optimal_volatility_var,
        'optimal_sharpe_ratio_var': optimal_sharpe_ratio_var,
        'var_5pct': portfolio_var,
        'cvar_5pct': portfolio_cvar,
        'expected_final_value': expected_final_value
    }


# Se vuoi eseguire la simulazione
if __name__ == "__main__":
    results = simulate_long_portfolio()