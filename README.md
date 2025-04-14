# Tesi di Laurea - Analisi delle Copule applicata al DAX

📘 Tesi di Laurea in Matematica per l’Ingegneria  
🎓 Politecnico di Torino, Anno Accademico 2024-2025  
👨‍🎓 Autore: Andrea Rostagno  
👨‍🏫 Relatori: Prof. 

---

## 📄 Titolo

**“Copule e Analisi della Dipendenza nei Rendimenti Finanziari: Applicazione all’indice DAX”**

---

## 🧠 Abstract

Questa tesi esplora l’utilizzo delle copule per modellare la dipendenza tra variabili finanziarie, concentrandosi sull’indice azionario tedesco DAX. Vengono analizzate diverse famiglie di copule (Gaussian, t-Student, Archimedean) e misure di dipendenza (Kendall's tau, Spearman's rho), sia dal punto di vista teorico che computazionale. 

Sono state condotte simulazioni Monte Carlo e fitting su dati storici reali del DAX, con implementazioni in Python. L’obiettivo è valutare la robustezza e l’adeguatezza di ciascun modello copula nell’ambito del risk management e del pricing di portafoglio.

---

## 🧪 Struttura del progetto

📁 DAX-Copula-Analysis-Thesis/ 
├── tesi_copule_DAX.pdf ← PDF completo della tesi 
├── TESI_PPTX.pptx ← Slide della presentazione 
├── formule.ipynb ← Notebook Jupyter con formule e codice Python 
├── /dati/ ← Dati storici del DAX (CSV) 
└── README.md ← Questo file


---

## 📚 Contenuti trattati

- Introduzione alle Copule
- Teorema di Sklar
- Copule Gaussiane e t-Student
- Famiglie Archimedeane (Clayton, Gumbel, Frank)
- Calibrazione su dati reali
- Simulazioni Monte Carlo
- Applicazione al portafoglio DAX
- Misure di dipendenza: τ di Kendall, ρ di Spearman
- Indicatori di goodness-of-fit

---

## 🐍 Requisiti (Python)

```bash
numpy
pandas
matplotlib
scipy
copulas
