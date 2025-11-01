# Adherence to the major classes of anthypertensive theraphy
This repository contains code for the supplementary nonparametric method in:

Karl Laurell, Stefan Gustafsson, Erik Lampa, Dave Zachariah, Sofia Ek, Karin Rådholm, Mats Martinell, Johan Sundström. 
"Initial drug choice for hypertension and long-term treatment persistence". 2025.

Data is not included in this repository. Dummy data is used for the examples.

### Study design
The study is an observational cohort study of patients starting antihypertensive therapy for the first time in Sweden. 
The study population consists of patients ≥ 40 years of age starting antihypertensive therapy for the first time in a 
single pill between 2011 and 2018. Individuals prescribed with an angiotensin receptor blocker (ARB), an angiotensin 
converting enzyme inhibitor (ACEi), a dihydropyridine calcium channel blocker (CCB), a thiazide/thiazide-like diuretic 
(TZD) or a single-pill combination (SPC) of those are included. In the main study, persistence on class and therapy 
level is determined. For the supplementary nonparametric method, the proportion of days adherent to treatment (PDAT) 
during the first year on class level is determined.

More information [here](https://catalogues.ema.europa.eu/node/3539/administrative-details).

### Results
The file main_pdat replicates the experiments for the supplementary nonparametric method using dummy data. 

The method is based on:
- Sofia Ek, Dave Zachariah, Fredrik D. Johansson, Petre Stoica. "Off-Policy Evaluation with Out-of-Sample Guarantees". 2023.

Using techniques to calibrate the IPW weights from:
- Sofia Ek, Dave Zachariah. "Externally Valid Policy Evaluation from Randomized Trials Using Additional Observational Data". 2024.






