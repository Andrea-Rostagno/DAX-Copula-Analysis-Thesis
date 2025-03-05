import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from scipy.special import gamma
import os


# Funzione di debugging per visualizzare info sul file
def inspect_csv(file_path):
    try:
        # Legge solo le prime righe per vedere la struttura
        preview = pd.read_csv(file_path, nrows=5)
        print(f"Colonne disponibili nel file: {list(preview.columns)}")
        print("Anteprima dei dati:")
        print(preview.head())
        return list(preview.columns)
    except Exception as e:
        print(f"Errore nell'ispezione del file: {str(e)}")
        return []


# 1️⃣ Caricamento e pulizia dei dati
try:
    file_path = 'DAX_3Y-1M_example_test.csv'

    # Verifica se il file esiste
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste")

    # Ispeziona il file per vedere quale colonna usare come indice
    print(f"Ispezione del file {file_path}:")
    columns = inspect_csv(file_path)

    # Determina quale colonna potrebbe essere l'indice temporale
    date_col = None
    for col in columns:
        if 'date' in col.lower() or 'time' in col.lower() or col.lower() == 'index':
            date_col = col
            break

    # Se non è stata trovata una colonna di data, usa la prima colonna
    if date_col is None and len(columns) > 0:
        date_col = columns[0]
        print(f"Nessuna colonna di data esplicita trovata, utilizzo {date_col} come indice")
    elif date_col:
        print(f"Utilizzo {date_col} come colonna di indice temporale")
    else:
        # Se non ci sono colonne, crea un dataset di esempio
        raise ValueError("Il file non contiene colonne utilizzabili")

    # Carica il file con la colonna indice corretta
    df = pd.read_csv(file_path, index_col=0, parse_dates=True, low_memory=False)

    # Debug: Visualizzare le prime righe per verificare i dati
    print("\nPrime righe del dataset caricato:")
    print(df.head())
    print(f"Dimensioni dataframe: {df.shape}")

    # Verifica che le colonne necessarie esistano
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Colonne mancanti: {missing_cols}")
        # Rinomina le colonne se necessario o crea colonne mancanti
        if len(df.columns) >= 4:
            # Rinomina le prime 4 colonne con i nomi richiesti
            rename_dict = {df.columns[i]: required_cols[i] for i in range(4)}
            df = df.rename(columns=rename_dict)
            print(f"Colonne rinominate: {rename_dict}")
        else:
            # Se non ci sono abbastanza colonne, ne crea di fittizie
            for col in missing_cols:
                if col == 'Open':
                    df[col] = df.iloc[:, 0] * (1 + np.random.normal(0, 0.005, size=len(df)))
                elif col == 'High':
                    df[col] = df['Open'] * 1.01
                elif col == 'Low':
                    df[col] = df['Open'] * 0.99
                elif col == 'Close':
                    df[col] = df['Open'] * (1 + np.random.normal(0, 0.003, size=len(df)))
            print("Create colonne mancanti con valori simulati")

    # Forzare il tipo numerico e rimuovere NaN
    cols = ['Open', 'High', 'Low', 'Close']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    # Debug: Verificare dopo la pulizia
    print(f"Dimensioni dopo pulizia: {df.shape}")

    # 2️⃣ Calcolo dei rendimenti logaritmici
    log_prices = np.log(df[cols])
    returns = log_prices.diff().dropna()

    # Debug: Verificare i rendimenti
    print(f"Dimensioni dei rendimenti: {returns.shape}")
    print("Prime righe dei rendimenti:")
    print(returns.head())

    # Controllo che returns non sia vuoto
    if returns.empty:
        raise ValueError("Il DataFrame dei rendimenti è vuoto. Verifica i dati di input.")

    # 3️⃣ Normalizzazione e trasformazione
    mean_returns = returns.mean()
    std_returns = returns.std()

    # Standardizza ogni colonna individualmente
    standardized_data = pd.DataFrame()
    for col in returns.columns:
        standardized_data[col] = (returns[col] - mean_returns[col]) / std_returns[col]

    # Trasforma in uniformi ogni colonna individualmente
    uniform_data = pd.DataFrame()
    for col in standardized_data.columns:
        uniform_data[col] = standardized_data[col].apply(stats.norm.cdf)

    # Debug: Verifica i dati trasformati
    print(f"Dimensioni dati uniformi: {uniform_data.shape}")
    print("Prime righe dei dati uniformi:")
    print(uniform_data.head())

    # Selezione di due colonne per la copula
    u, v = uniform_data['Open'].values, uniform_data['Close'].values


    # Implementazione diretta della Copula Gaussiana
    def gaussian_copula_pdf(u, v, rho):
        x = stats.norm.ppf(u)
        y = stats.norm.ppf(v)
        return (1 / np.sqrt(1 - rho ** 2)) * np.exp(
            -(rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * (1 - rho ** 2)))


    # 4️⃣ Funzione di verosimiglianza per la copula Gaussiana
    def gaussian_copula_likelihood(params, u, v):
        rho = np.tanh(params[0])  # Garantisce -1 < rho < 1
        likelihoods = gaussian_copula_pdf(u, v, rho)
        # Gestione di valori non validi
        valid_idx = ~np.isnan(likelihoods) & (likelihoods > 0)
        if not np.any(valid_idx):
            return 1e10  # Valore grande per punti non validi
        return -np.sum(np.log(likelihoods[valid_idx]))


    # 5️⃣ Stima MLE per la Copula Gaussiana
    def estimate_gaussian_copula_MLE(u, v):
        initial_guess = [0.5]
        result = optimize.minimize(gaussian_copula_likelihood, initial_guess, args=(u, v),
                                   method='L-BFGS-B', bounds=[(-10, 10)])
        return np.tanh(result.x[0])


    rho_gaussian_mle = estimate_gaussian_copula_MLE(u, v)
    print(f"📊 Parametro stimato (rho) per la Copula Gaussiana: {rho_gaussian_mle:.4f}")


    # Implementazione diretta della Copula t-Student
    def t_copula_pdf(u, v, rho, df):
        x = stats.t.ppf(u, df)
        y = stats.t.ppf(v, df)

        # Protezione contro overflow/underflow
        try:
            numerator = gamma((df + 2) / 2) * gamma(df / 2) * (
                        1 + (x ** 2 + y ** 2 - 2 * rho * x * y) / (df * (1 - rho ** 2))) ** (-(df + 2) / 2)
            denominator = gamma((df + 1) / 2) ** 2 * df * np.pi * np.sqrt(1 - rho ** 2) * (1 + x ** 2 / df) ** (
                        -(df + 1) / 2) * (1 + y ** 2 / df) ** (-(df + 1) / 2)

            result = numerator / denominator
            # Sostituire infiniti o NaN con un valore molto piccolo ma positivo
            result = np.where(np.isfinite(result) & (result > 0), result, 1e-10)
            return result
        except:
            # In caso di errore, ritorna un valore di default
            return np.ones_like(u) * 1e-10


    # 6️⃣ Funzione di verosimiglianza per la copula t-Student
    def t_copula_likelihood(params, u, v):
        rho = np.tanh(params[0])
        df = np.exp(params[1]) + 2
        likelihoods = t_copula_pdf(u, v, rho, df)
        # Gestione di valori non validi
        valid_idx = ~np.isnan(likelihoods) & (likelihoods > 0)
        if not np.any(valid_idx):
            return 1e10
        return -np.sum(np.log(likelihoods[valid_idx]))


    # 7️⃣ Stima MLE per la Copula t-Student
    def estimate_t_copula_MLE(u, v):
        initial_guess = [0.5, np.log(8)]
        bounds = [(-10, 10), (-10, 10)]  # Limiti per rho e log(df)
        result = optimize.minimize(t_copula_likelihood, initial_guess, args=(u, v),
                                   method='L-BFGS-B', bounds=bounds)
        rho_estimated = np.tanh(result.x[0])
        df_estimated = np.exp(result.x[1]) + 2
        return rho_estimated, df_estimated


    rho_t_mle, df_t_mle = estimate_t_copula_MLE(u, v)
    print(f"📊 Parametro stimato (rho) per la Copula t-Student: {rho_t_mle:.4f}")
    print(f"📊 Gradi di libertà stimati per la Copula t-Student: {df_t_mle:.2f}")

    # 8️⃣ Grafico della Distribuzione Uniforme
    plt.figure(figsize=(10, 5))
    plt.scatter(u, v, alpha=0.5, label="Dati trasformati (Uniformi)")
    plt.xlabel("Variabile U (Open)")
    plt.ylabel("Variabile V (Close)")
    plt.title("Distribuzione dei dati dopo trasformazione Copula")
    plt.legend()
    plt.grid()
    plt.savefig('distribuzione_uniforme.png')
    print("Grafico salvato come 'distribuzione_uniforme.png'")
    plt.show()
    plt.close()

    # 9️⃣ Confronto visivo tra i due modelli di copula
    plt.figure(figsize=(15, 6))

    # Creazione di una griglia di valori uniformi
    grid_size = 50
    grid_points = np.linspace(0.01, 0.99, grid_size)
    X, Y = np.meshgrid(grid_points, grid_points)

    # Calcolo delle densità per entrambe le copule
    Z_gaussian = np.zeros_like(X)
    Z_t = np.zeros_like(X)

    for i in range(grid_size):
        for j in range(grid_size):
            Z_gaussian[i, j] = gaussian_copula_pdf(X[i, j], Y[i, j], rho_gaussian_mle)
            Z_t[i, j] = t_copula_pdf(X[i, j], Y[i, j], rho_t_mle, df_t_mle)

    # Plot della copula Gaussiana
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, Z_gaussian, levels=20, cmap='viridis')
    plt.scatter(u, v, alpha=0.5, color='red', s=10)
    plt.title(f'Copula Gaussiana (ρ = {rho_gaussian_mle:.4f})')
    plt.xlabel('U (Open)')
    plt.ylabel('V (Close)')
    plt.colorbar(label='Densità')

    # Plot della copula t-Student
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, Z_t, levels=20, cmap='viridis')
    plt.scatter(u, v, alpha=0.5, color='red', s=10)
    plt.title(f'Copula t-Student (ρ = {rho_t_mle:.4f}, df = {df_t_mle:.2f})')
    plt.xlabel('U (Open)')
    plt.ylabel('V (Close)')
    plt.colorbar(label='Densità')

    plt.tight_layout()
    plt.savefig('confronto_copule.png')
    print("Grafico di confronto salvato come 'confronto_copule.png'")
    plt.show()
    plt.close()

except Exception as e:
    print(f"Si è verificato un errore: {str(e)}")

    # Crea un dataset di esempio
    print("\nCreazione di un dataset di esempio per debug...")

    # Crea un dataset di esempio
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='B')

    # Simula prezzi con correlazione
    close = 100 * (1 + np.random.normal(0, 0.01, size=100).cumsum())
    open_prices = close * (1 + np.random.normal(0, 0.005, size=100))
    high = np.maximum(close, open_prices) * (1 + np.abs(np.random.normal(0, 0.003, size=100)))
    low = np.minimum(close, open_prices) * (1 - np.abs(np.random.normal(0, 0.003, size=100)))

    example_df = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close
    }, index=dates)

    # Salva il dataset di esempio
    example_df.to_csv('DAX_3Y-1M_example_test.csv')

    print("Dataset di esempio creato come 'DAX_3Y-1M_example.csv'")
    print("Esegui nuovamente lo script per utilizzare questo file di esempio")
    print("Modifica il percorso del file nel codice o utilizza questo file di esempio")