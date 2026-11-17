#set document(title: "How Much of a Backtest Is Luck? Selection Bias in a 1,979-Model Equity Strategy Search")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11pt)
#show figure.caption: set text(size: 9pt)
#show figure: set block(breakable: false)
#let fig(n, name, w, cap) = {
  counter(figure.where(kind: image)).update(n - 1)
  [#figure(image("figures/" + name + ".png", width: w), caption: cap, placement: auto)#label("fig" + str(n))]
}

#align(center)[
  #text(size: 17pt, weight: "bold")[How Much of a Backtest Is Luck?] \
  #v(2pt)
  #text(size: 12pt)[Selection bias in a 1,979-model equity strategy search]
]
#v(10pt)

#pad(x: 1.2cm)[
  *Abstract.* A search over 1,979 machine-learning configurations for a cross-sectional equity strategy
  (758 gradient-boosted tree models and 1,221 neural networks) produced a best net Sharpe ratio of 1.03 over 249 monthly periods.
  This paper applies the standard diagnostics for selection bias to the complete set of trials: the probability of backtest overfitting (PBO) by
  combinatorially symmetric cross-validation, the deflated Sharpe ratio, multiple-testing haircuts, and White's Reality Check and Hansen's test for superior predictive ability.
  The diagnostics are first validated on synthetic trial sets with a known answer. The trials share one signal: every configuration has a positive net Sharpe ratio,
  the best configuration retains a Sharpe ratio of 0.62 after a Bonferroni or Holm haircut for 1,979 tests, and the deflated Sharpe ratio is 0.999.
  The selection among configurations is not informative: PBO is 0.44, the in-sample winner has an out-of-sample Sharpe ratio of 0.77 against 0.78 for the average configuration,
  and searching more configurations raised the best reported Sharpe ratio only from 0.91 (10 trials) to 1.03.
  The universe is survivorship-biased, so absolute performance is overstated; the conclusions concern the selection problem.
]

= Introduction

A backtest reports the performance of one strategy, but the strategy is usually the best of many that were tried. The expected maximum of a set of noisy performance estimates
increases with the size of the set, so the best Sharpe ratio is a biased estimate of that strategy's true Sharpe ratio. Two questions follow. First, is there any skill in the set at all?
Second, if the best configuration is chosen and deployed, will it outperform the others out of sample? These are different questions, and a single search can give different answers to each.

This paper answers both for one concrete search. The search was run by an automated research pipeline that fits cross-sectional return forecasts for large-cap US equities with
randomly drawn hyper-parameters and records the return series of every trial. Because every trial was kept, the complete set can be analysed, which is rarely possible with published backtests.
The contributions are (i) an exact and vectorised implementation of the diagnostics, tested against slow reference code and an independent implementation;
(ii) a Monte Carlo validation on synthetic data with a planted signal; and (iii) the application to 1,979 real trials, with sensitivity analyses.

= Data and the search <sec-data>

== Trials

The trials come from a walk-forward procedure. For each calendar year, each model is refitted on all earlier data, with a gap of 40 calendar days between the end of training and the start of the test year,
and predicts the next 21-trading-day return of every stock in the cross-section. On each rebalance date, spaced 21 trading days apart, the strategy buys the decile of stocks with the highest predictions and sells the decile
with the lowest. The cost model charges 10 basis points per trade, applied to the sale of each departing stock and the purchase of each entering stock in both legs.
The sample has 249 rebalance dates from 2006-01-06 to 2026-09-22. Hyper-parameters were drawn at random from a grid: 758 LightGBM configurations (learning rate, leaves, minimum leaf size,
feature and bagging fractions, L2 penalty, number of trees, objective) and 1,221 multilayer-perceptron configurations (width, depth, dropout, learning rate, weight decay, epochs, batch size). Each draw also has a random seed.

== Return matrix

Net returns of all trials form a matrix $R in RR^(T times N)$ with $T = 249$ and $N = 1,979$. Sharpe ratios are computed from per-period returns with the sample standard deviation and annualised by $sqrt(252 / 21)$.
The recomputed Sharpe ratio of every trial matches the value logged by the search pipeline to within $10^(-6)$.
The Sharpe ratios of the trials have mean 0.78, median 0.80, standard deviation 0.12, minimum 0.12 and maximum 1.03. The trials are strongly correlated (@fig8):
the mean pairwise correlation of net returns is 0.87 and the first principal component carries 87 % of the total variance.

#fig(8, "fig8_correlation", 50%, [*Correlation structure of the trials.* Correlation of net returns between trials, ordered by hierarchical clustering and averaged in 150 × 150 blocks. The mean pairwise correlation is 0.87 and the first principal component carries 87 % of the variance.])

= Methods

== Probability of backtest overfitting

Following Bailey, Borwein, López de Prado and Zhu @bailey2017pbo, the $T$ periods are divided into $S = 16$ contiguous blocks of equal length (the first $T mod S = 9$ periods are dropped, leaving blocks of 15 periods).
Each of the $binom(16, 8) = 12,870$ ways of choosing eight blocks defines a training set; the other eight blocks form the test set. On each split the trial $n^*$ with the highest training Sharpe ratio is selected,
and its relative rank $omega = r / (N + 1)$ among all $N$ trials on the test set is converted to the logit $lambda = ln(omega / (1 - omega))$. The probability of backtest overfitting is
$ "PBO" = P(lambda <= 0), $
the share of splits in which the selected trial ranks at or below the median out of sample. All $N$ Sharpe ratios for all splits are obtained from block sums and sums of squares with one matrix product, which is exact.
Also reported are the slope of a regression of out-of-sample on in-sample Sharpe ratio of the selected trial (performance degradation), the share of splits with a negative out-of-sample Sharpe ratio, and first- and second-order stochastic dominance of the
selected trial's out-of-sample distribution over that of the average trial.

== Probabilistic and deflated Sharpe ratio

With per-period Sharpe ratio $hat(S R)$, skewness $hat(gamma)_3$ and kurtosis $hat(gamma)_4$ estimated from $T$ observations, the probabilistic Sharpe ratio is
$ "PSR"(S R^*) = Phi( ((hat(S R) - S R^*) sqrt(T - 1)) / sqrt(1 - hat(gamma)_3 hat(S R) + (hat(gamma)_4 - 1) / 4 hat(S R)^2) ) $
@bailey2012sre @mertens2002. The expected maximum Sharpe ratio of $N$ independent trials with no skill and cross-trial variance $V$ is approximated by
$ S R_0 = sqrt(V) ( (1 - gamma_E) Phi^(-1)(1 - 1 / N) + gamma_E Phi^(-1)(1 - 1 / (N e)) ), $
where $gamma_E$ is the Euler-Mascheroni constant, and the deflated Sharpe ratio is $"DSR" = "PSR"(S R_0)$ @bailey2014dsr. The minimum track record length is the smallest $T$ for which $"PSR"(0)$ reaches 0.95.
Correlation among trials lowers the effective number of independent trials. Three diagnostics are reported: the eigenvalue participation ratio, the number of clusters from average-linkage clustering at several correlation cut-offs,
and a calibrated count $N^*$, defined as the number of independent trials for which the expression above reproduces the expected maximum of a simulated zero-skill null. The null demeans every trial and resamples time jointly with the stationary bootstrap,
which preserves the cross-sectional correlation.

== Multiple-testing haircuts

Following Harvey and Liu @harvey2015backtesting, the two-sided p-value of each Sharpe ratio is $p = 2 (1 - Phi(|t|))$ with $t = hat(S R) sqrt(T)$. Adjusted p-values use the Bonferroni, Holm and Benjamini-Hochberg-Yekutieli
@benjamini2001bhy procedures. An adjusted p-value is converted back to a t-statistic and an annualised Sharpe ratio; the haircut is one minus the ratio of the adjusted to the raw Sharpe ratio.

== Reality Check and test for superior predictive ability

White's Reality Check @white2000reality and Hansen's SPA test @hansen2005spa test the null that no trial has a positive expected return, against a zero-return benchmark. Resamples use the stationary bootstrap of Politis and Romano
@politis1994stationary with 100,000 resamples; all trials are evaluated in one matrix product per chunk of resamples. The SPA test studentises each trial and recentres trials with clearly negative means (the "consistent" variant).
The mean block length was set to 2 after a size check on synthetic data (@sec-syn).

= Synthetic validation <sec-syn>

The diagnostics were run on synthetic trial sets with $T = 249$ periods. In the first design, 300 independent zero-mean series are combined with one series that has a planted annual Sharpe ratio. In the second, 15 families of 20 series share a common factor
(within-family correlation 0.8) and one family carries the planted Sharpe ratio. Each configuration was repeated 100 times (@fig7).

With no planted signal, mean PBO is 0.49 and the share of runs with DSR above 0.95 is 0.00. The SPA test rejects at the 5 % level in 0.06 of runs and the Reality Check in 0.05.
At a planted annual Sharpe ratio of 1.0, mean PBO is 0.20 and DSR exceeds 0.95 in 0.43 of runs; at 1.5 the values are 0.025 and 0.98; at 2.0 they are 0.000 and 1.00.
With 249 monthly periods and 300 trials, a planted Sharpe ratio of 1.0 is therefore not reliably identified: the best noise trial exceeds the planted one in-sample in a substantial share of splits.
In the correlated-family design, mean PBO is 0.50 without signal and 0.06 at a planted Sharpe ratio of 1.0.
Mean block length also matters for the SPA test: in a size experiment on 300 independent null trials (150 replications), the SPA test rejected at the 5 % level in 3 % of runs with mean block length 1, 5 % with 2, 8 % with 3 and 18 % with 6, because a long block makes the bootstrap variance of each trial noisy.
The reported results use a mean block length of 2.

#fig(7, "fig7_synthetic", 100%, [*Synthetic validation.* Mean PBO, share of runs with a deflated Sharpe ratio above 0.95, and SPA rejection rate at the 5 % level, as a function of the planted annual Sharpe ratio. Blue: 300 independent trials. Red: 15 families of 20 trials with within-family correlation 0.8. T = 249 in all cases.])

= Results

== Evidence of skill

The best configuration (a neural network) has a net Sharpe ratio of 1.03, skewness 1.04 and kurtosis 6.3. Its minimum track record length is 26 periods.
With the expected maximum under no skill at 0.43 (threshold from the observed cross-trial variance), the deflated Sharpe ratio is 0.999. Using the simulated zero-skill maximum of 0.29 instead gives 0.9999.
The raw two-sided p-value of the best trial is 2.4e-06; after a Bonferroni or Holm adjustment for 1,979 tests it is 0.0048, and the haircut Sharpe ratio is 0.62 (a haircut of 40 %).
The Benjamini-Hochberg-Yekutieli adjustment gives 0.65 (37 %). Of the 1,979 trials, 105 remain significant at the 5 % level under Bonferroni, 116 under Holm and 1,797 under BHY.
The Reality Check and SPA tests reject the null of no skill (Reality Check p < 0.001, SPA p < 0.001; all mean block lengths from 1 to 12 give p < 0.001). @fig6 summarises the adjustments.

#fig(6, "fig6_ladder", 88%, [*Sharpe ratio after each adjustment.* Annualised Sharpe ratio of the selected trial after multiple-testing haircuts and after subtracting the deflation thresholds, with the in-sample and out-of-sample CSCV averages for the selected trial and for the average trial.])

The expected maximum Sharpe ratio under no skill is lower than the observed one at every search size (@fig5, left): 0.11 against 0.91 for 10 trials and 0.32 against 1.03 for 1,979.
The effective number of independent trials is not materially below the nominal one in the calibrated sense (1,979), although the correlation structure is strong: the participation ratio is 1.3, 4 eigenvalues exceed the Marchenko-Pastur edge,
and clustering at a correlation of 0.9 yields 157 groups. The strong common component raises the Sharpe ratio of all trials together but does not reduce the number of idiosyncratic draws available to the maximum.

#fig(5, "fig5_search_size", 100%, [*Effect of the size of the search.* Left: best full-sample Sharpe ratio among n randomly drawn trials (blue) and the expected best Sharpe ratio of n trials with zero skill and the same correlation structure (red); bands are 5th to 95th percentiles. Right: PBO of the same subsets. The best observed Sharpe ratio rises from 0.91 to 1.03 as n grows from 10 to 1,979, while PBO stays between 0.37 and 0.44.])

== Evidence on selection

PBO is 0.444 across 12,870 splits (@fig2). The selected trial has a mean in-sample Sharpe ratio of 1.15 and a mean out-of-sample Sharpe ratio of 0.77, against 0.78 for the average trial.
The slope of out-of-sample on in-sample Sharpe ratio is -0.50 (@fig1). The share of splits with a negative out-of-sample Sharpe ratio is 0.000, so PBO here measures the loss of rank, not a loss of money.
The selected trial is not a single configuration: 218 distinct trials are selected, the most frequent in 10.6 % of splits. PBO is 0.29 within the LightGBM trials and 0.39 within the neural networks.
With S = 12 and S = 20 blocks PBO is 0.44 and 0.44; with S = 8, which gives only 70 splits, it is 0.26.
Searching more trials does not change PBO materially (@fig5, right): the mean over random subsets lies between 0.37 and 0.44 for every subset size from 10 to 1,979.
@fig3 shows the joint density of in-sample and out-of-sample rank: the weakest trials are weak in both halves, while the upper part of the ranking is close to independent.
@fig4 shows the hyper-parameter landscape.

#fig(1, "fig1_is_vs_oos", 50%, [*In-sample versus out-of-sample Sharpe ratio of the selected trial.* Annualised Sharpe ratio of the trial with the highest in-sample Sharpe ratio against its Sharpe ratio on the held-out blocks, for all 12,870 CSCV splits (S = 16). The fitted slope is -0.50: splits with a higher in-sample value have a lower out-of-sample value. Part of this slope is mechanical, because the two halves of a split partition one fixed sample.])
#fig(2, "fig2_logit_hist", 52%, [*Distribution of the logit of the out-of-sample rank.* Logit of the relative out-of-sample rank of the selected trial across 12,870 splits. Splits with λ ≤ 0 (shaded) are those in which the selected trial ranks at or below the median out of sample; their share is the PBO, 0.444.])
#fig(3, "fig3_rank_surface", 46%, [*Joint density of in-sample and out-of-sample rank.* Density of a trial's in-sample rank percentile and out-of-sample rank percentile over all 1,979 trials and all splits, relative to independence (log base 2, floored at 2^-6). The mass concentrated at low ranks on both axes shows that the weakest trials are consistently weak, while the upper part of the ranking is close to independent between the two halves.])
#fig(4, "fig4_landscape", 100%, [*Hyper-parameter landscape.* Mean out-of-sample Sharpe ratio of each trial, plotted against the two hyper-parameters most rank-correlated with it within each model family. Small random offsets separate overlapping grid points.])

#figure(
  table(columns: (auto, auto), align: (left, right), stroke: (x: none, y: 0.4pt + gray),
    [*Quantity*], [*Value*],
    [Trials (LightGBM / neural network)], [1,979 (758 / 1,221)],
    [Best net Sharpe ratio], [1.03],
    [Mean Sharpe ratio of all trials], [0.78],
    [Deflated Sharpe ratio (nominal $N$)], [0.999],
    [Haircut Sharpe ratio, Bonferroni / Holm], [0.62],
    [Haircut Sharpe ratio, BHY], [0.65],
    [Reality Check p-value], [< 0.001],
    [SPA p-value (consistent)], [< 0.001],
    [PBO (S = 16)], [0.44],
    [Selected trial, in-sample / out-of-sample Sharpe], [1.15 / 0.77],
    [Average trial, out-of-sample Sharpe], [0.78],
    [Degradation slope], [-0.50],
  ),
  caption: [Summary of results. Sharpe ratios are annualised and net of costs.],
) <tab1>

= Discussion and limitations

The results separate two properties of the search. The set of trials contains skill: all configurations are profitable, the best one survives every adjustment applied here, and the null of no skill is rejected by tests that account for the full search.
The ranking of configurations within the set carries little information. The best in-sample configuration does no better out of sample than a typical one, and PBO of 0.44 means that its out-of-sample rank is below the median in a little under half of the splits.
In practice this suggests that the choice among similar configurations of one model family should be made on grounds other than the maximum of the in-sample Sharpe ratio, for example by averaging forecasts across configurations.

Limitations:
- *Survivorship.* The universe consists of current index members, so absolute returns and Sharpe ratios are overstated. The relative conclusions about selection are less affected, but the levels should not be read as achievable performance.
- *One search.* All trials come from one pipeline, one feature set and one sample period. The results are not evidence about other searches.
- *Dependence and non-normality.* Sharpe ratios are annualised assuming independent periods, but the mean first-order autocorrelation of trial returns is -0.24, which makes the annualised ratios conservative.
  The p-values in the haircut calculation are normal approximations and the synthetic validation uses Gaussian returns.
- *CSCV construction.* The two halves of each split partition one fixed sample, which produces a negative relation between in-sample and out-of-sample Sharpe ratios even without overfitting, so the slope in @fig1 should be read together with the rank-based PBO.
  PBO also depends on the number of blocks; S = 8 gives a lower value than S $>=$ 12.
- *No untouched holdout.* The same periods are used to select a configuration and to evaluate it; CSCV mitigates but does not replace a holdout.
- *Effective number of trials.* The calibrated count equals the nominal count here, while cluster counts depend on the cut-off and range from 2 to 157.
- *Validation against the literature.* The deflated Sharpe ratio was checked against closed-form values, a Monte Carlo of the expected maximum and an independent implementation (agreement to $10^(-13)$), and the SPA test against the `arch` package (p-values within 0.03).
  Numerical examples from the original papers were not reproduced.

#bibliography("refs.bib", style: "apa", title: "References")
