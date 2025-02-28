# OSM-delta-gamma-hedging
Implementation for
>  B.Negyesi, C.W.Oosterlee (2025): A Deep BSDE approach for the simultaneous pricing and delta-gamma hedging of large portfolios consisting of high-dimensional multi-asset Bermudan options
> https://arxiv.org/abs/2502.11706v1

## Usage
### collect deep BSDE approximations
1. create setup in a configuration file in `/configs/`: `example.json`
2. run `python main_rOSM_portfolio.py example.json run_name`
3. deep BSDE approximations are collected under `/logs/systemOSM_theta...>` depending on the theta parameter of the setup

Similarly for deep BSDE approximations given by 
>C. Huré, H. Pham, & X. Warin (2020): Deep backward schemes for high-dimensional nonlinear PDEs. Mathematics of Computation, 89(324), 1547–1579. https://doi.org/10.1090/mcom/3514

### hedging
1. run `sample_assets.py` with the correct `run_name`
2. Delta hedging: run `delta_hedging_portfolio_black_scholes.py`
3. Delta-Gamma hedging:
   1. collect hedging instruments: `gather_hedging_instruments.py`
   2. compute hedging weights, and pnl sample: `gamma_hedging_sparse.py`

similarly for stochastic volatility examples

## FBSDEs
new FBSDE systems can be implemented by
- inheriting from the base class `SystemReflectedFBSDE` in `/lib/equations/portfolios.py`
- implementing the asssociated coefficients

Default approximation of the corresponding SDEs is given by the Euler-Maruyama schemes eq. (8)-(17) in the paper.