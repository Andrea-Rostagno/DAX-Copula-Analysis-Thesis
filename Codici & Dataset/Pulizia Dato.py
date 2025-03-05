import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Parte 1: Diagnostica del file CSV
print("🔍 DIAGNOSTICA DEL FILE CSV")
print("-" * 50)

csv_path = 'DAX_3Y-1M.csv'

# Verificare se il file esiste
if not os.path.exists(csv_path):
    print(f"⚠️ ERRORE: Il file '{csv_path}' non esiste nella directory corrente.")
    print(f"Directory corrente: {os.getcwd()}")
    print(f"File nella directory: {os.listdir()}")
else:
    print(f"✅ File '{csv_path}' trovato.")

    # Ispeziona le prime righe del file
    print("\n📄 Ispeziono le prime righe del file:")
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i < 10:  # Mostra solo le prime 10 righe
                print(f"Riga {i + 1}: {line.strip()}")
            else:
                break

    # Identifica il delimitatore
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        sample = f.read(1024)
        dialect = csv.Sniffer().sniff(sample)
        print(f"\n📊 Delimitatore rilevato: '{dialect.delimiter}'")
        has_header = csv.Sniffer().has_header(sample)
        print(f"📋 Intestazione rilevata: {has_header}")

    # Prova a leggere con pandas utilizzando diversi delimitatori
    delimiters = [',', ';', '\t', '|']
    for delimiter in delimiters:
        try:
            df_sample = pd.read_csv(csv_path, sep=delimiter, nrows=5)
            print(f"\n✅ Lettura riuscita con delimitatore '{delimiter}':")
            print(df_sample.head())
            found_delimiter = delimiter
            break
        except Exception as e:
            print(f"\n❌ Delimitatore '{delimiter}' non funziona: {str(e)}")

    # Tentativo di lettura con varie configurazioni
    print("\n🧪 Probo diverse configurazioni per la lettura del file...")

    try:
        df = pd.read_csv(csv_path, sep=found_delimiter, encoding='utf-8')
        print("✅ Lettura base riuscita")
        print(f"Dimensioni: {df.shape}")
        print(f"Colonne: {df.columns.tolist()}")
        print(f"Prime 5 righe:\n{df.head()}")
    except Exception as e:
        print(f"❌ Errore nella lettura base: {str(e)}")

    # Tento con diverse configurazioni di encoding e impostazioni
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, sep=found_delimiter, encoding=encoding,
                             error_bad_lines=False, warn_bad_lines=True)
            print(f"\n✅ Lettura riuscita con encoding '{encoding}'")
            if 'DateTime' in df.columns:
                print("✅ Colonna 'DateTime' trovata")
            else:
                print("⚠️ Colonna 'DateTime' non trovata")
                print(f"   Colonne disponibili: {df.columns.tolist()}")

            # Verifichiamo se le colonne numeriche sono presenti
            expected_cols = ['Open', 'High', 'Low', 'Close']
            found_cols = [col for col in expected_cols if col in df.columns]

            if found_cols:
                print(f"✅ Trovate colonne numeriche: {found_cols}")
                numeric_df = df[found_cols]
                print(f"   Info sui tipi di dati:\n{numeric_df.dtypes}")

                # Verifica per valori non numerici
                non_numeric = {}
                for col in found_cols:
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except Exception as e:
                        mask = pd.to_numeric(df[col], errors='coerce').isna()
                        non_numeric[col] = df.loc[mask, col].unique().tolist()[:5]  # Primi 5 valori problematici

                if non_numeric:
                    print(f"⚠️ Trovati valori non numerici in: {list(non_numeric.keys())}")
                    for col, values in non_numeric.items():
                        print(f"   Colonna '{col}' - esempi: {values}")
                else:
                    print("✅ Tutti i valori nelle colonne numeriche sono validi")
            else:
                print(f"⚠️ Nessuna delle colonne numeriche attese trovate: {expected_cols}")

            break  # Se la lettura ha successo, esce dal ciclo

        except Exception as e:
            print(f"❌ Errore con encoding '{encoding}': {str(e)}")

# Parte 2: Correzione e preprocessing
print("\n\n🛠️ CORREZIONE E PREPROCESSING DEI DATI")
print("-" * 50)

try:
    # Tentativo di lettura con le impostazioni ottimali
    best_encoding = 'utf-8'  # Da aggiornare in base ai risultati della diagnosi
    best_delimiter = ','  # Da aggiornare in base ai risultati della diagnosi

    df = pd.read_csv(csv_path, sep=best_delimiter, encoding=best_encoding,
                     low_memory=False, na_values=['na', 'NA', 'N/A', ''])

    # Verifica e correzione della colonna datetime
    if 'DateTime' in df.columns:
        try:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            print("✅ Convertita colonna 'DateTime' in formato datetime")
        except Exception as e:
            print(f"⚠️ Impossibile convertire 'DateTime': {str(e)}")

            # Tentativo di correzione del formato della data
            print("   Tentativo di correzione del formato della data...")

            # Mostra esempi di date per diagnostica
            date_samples = df['DateTime'].head(5).tolist()
            print(f"   Esempi di date: {date_samples}")

            # Prova diversi formati di parsing
            date_formats = ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']
            for date_format in date_formats:
                try:
                    df['DateTime'] = pd.to_datetime(df['DateTime'], format=date_format)
                    print(f"✅ Convertita 'DateTime' con formato '{date_format}'")
                    break
                except:
                    continue
    else:
        print("⚠️ Colonna 'DateTime' non trovata")

        # Cerca possibili alternative
        datetime_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_candidates:
            print(f"   Candidati alternativi per datetime: {datetime_candidates}")
            # Prova a rinominare il primo candidato
            df.rename(columns={datetime_candidates[0]: 'DateTime'}, inplace=True)
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                print(f"✅ Utilizzato '{datetime_candidates[0]}' come 'DateTime'")
            except:
                print(f"❌ Impossibile convertire '{datetime_candidates[0]}' in datetime")

    # Verifica e correzione delle colonne numeriche
    price_cols = ['Open', 'High', 'Low', 'Close']
    available_cols = [col for col in price_cols if col in df.columns]

    if available_cols:
        print(f"\n📊 Convertendo colonne di prezzo: {available_cols}")

        # Verifica tipi di dati attuali
        print(f"   Tipi di dati attuali:\n{df[available_cols].dtypes}")

        # Converti in float
        for col in available_cols:
            try:
                # Prima verifica se ci sono problemi di formato (es. numeri separati da virgole)
                if df[col].dtype == 'object':
                    # Controlla se i numeri usano virgola come separatore decimale
                    if any(',' in str(x) for x in df[col].dropna().head(10)):
                        df[col] = df[col].str.replace(',', '.').astype(float)
                        print(f"   ✅ Colonna '{col}': convertita da formato con virgole")
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"   ✅ Colonna '{col}': convertita in numerica")
                else:
                    print(f"   ✓ Colonna '{col}' già in formato numerico: {df[col].dtype}")
            except Exception as e:
                print(f"   ❌ Errore nella conversione di '{col}': {str(e)}")

        # Rimuovi le righe con valori mancanti
        rows_before = len(df)
        df.dropna(subset=available_cols, inplace=True)
        rows_after = len(df)
        print(f"\n🧹 Rimosse {rows_before - rows_after} righe con valori mancanti")
    else:
        print(f"⚠️ Nessuna delle colonne di prezzo attese trovate: {price_cols}")

        # Cerca possibili alternative
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"   Colonne numeriche disponibili: {numeric_cols}")

        if len(numeric_cols) >= 4:
            # Rinomina le prime 4 colonne numeriche
            rename_dict = {numeric_cols[i]: price_cols[i] for i in range(4)}
            df.rename(columns=rename_dict, inplace=True)
            print(f"   ✅ Rinominate colonne: {rename_dict}")
            available_cols = price_cols[:4]
        elif len(numeric_cols) > 0:
            print("   ⚠️ Numero insufficiente di colonne numeriche per il modello completo")

    # Visualizza statistiche di base
    if available_cols:
        print("\n📈 Statistiche sui dati numerici:")
        print(df[available_cols].describe())

        # Controlla per outlier
        for col in available_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            print(f"   Colonna '{col}': {outliers} potenziali outlier ({outliers / len(df) * 100:.2f}%)")

    # Imposta l'indice come DateTime se disponibile
    if 'DateTime' in df.columns:
        df.set_index('DateTime', inplace=True)
        print("\n🕒 Impostato 'DateTime' come indice")

    # Salva i dati puliti
    clean_csv_path = 'DAX_cleaned.csv'
    df.to_csv(clean_csv_path)
    print(f"\n💾 Dati puliti salvati in '{clean_csv_path}'")
    print(f"    Dimensioni finali: {df.shape}")

    # Visualizzazione dati
    if available_cols and len(df) > 0:
        print("\n📊 Visualizzazione dei dati puliti:")
        print(df.head())

        plt.figure(figsize=(12, 6))
        for col in available_cols[:1]:  # Mostra solo il primo per semplicità
            plt.plot(df.index[-100:], df[col][-100:], label=col)
        plt.title('Ultimi 100 valori di prezzo')
        plt.xlabel('Data')
        plt.ylabel('Prezzo')
        plt.legend()
        plt.grid(True)
        plt.savefig('price_chart.png')
        print("\n📊 Grafico salvato come 'price_chart.png'")

except Exception as e:
    print(f"\n❌ Errore durante la correzione: {str(e)}")

print("\n✅ Diagnostica e correzione completate!")