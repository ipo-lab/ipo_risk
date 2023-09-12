# ipo_risk
This repository provides proof-of-concept experiments for integrated covariance estimation with risk-based portfolio optimization.
For more details please see our paper:

[Paper](https://www.risk.net/journal-of-risk/7905781/covariance-estimation-for-risk-based-portfolio-optimization-an-integrated-approach)


## Core Dependencies:
To run the experiments you will need to install:
* [numpy](https://numpy.org)
* [Pytorch](https://pytorch.org)
* [lqp_py](https://github.com/ipo-lab/lqp_py): an efficient batch QP solver.
* [rpth](https://github.com/ipo-lab/rpth): an efficient batch solver for risk-parity portfolios.

Please see requirements.txt for full details.

## Proof-of-concept experiments:
Proof-of-concept experiments are available in the [experiments](experiments) directory and summarized below.

### Minimum Variance Portfolio:
#### Constant Correlation and FF3 factors:
Out-of-sample Variance     |  Equity
:-------------------------:|:-------------------------:
![minvar_ccc_ff3](/images/minvar_ccc_ff3.png)  |  ![minvar_ccc_ff3_equity](/images/minvar_ccc_ff3_equity.png)

#### Constant Correlation and FF5 factors:
Out-of-sample Variance     |  Equity
:-------------------------:|:-------------------------:
![minvar_ccc_ff5](/images/minvar_ccc_ff5.png)  |  ![minvar_ccc_ff5_equity](/images/minvar_ccc_ff5_equity.png)

#### Dynamic Correlation and FF3 factors:
Out-of-sample Variance     |  Equity
:-------------------------:|:-------------------------:
![minvar_dcc_ff3](/images/minvar_dcc_ff3.png)  |  ![minvar_dcc_ff3_equity](/images/minvar_dcc_ff3_equity.png)

#### Dynamic Correlation and FF5 factors:
Out-of-sample Variance     |  Equity
:-------------------------:|:-------------------------:
![minvar_dcc_ff5](/images/minvar_dcc_ff5.png)  |  ![minvar_dcc_ff5_equity](/images/minvar_dcc_ff5_equity.png)


### Maximum Diversification Portfolio:
#### Constant Correlation and FF3 factors:
Out-of-sample Diversification |  Equity
:-------------------------:|:-------------------------:
![maxdiv_ccc_ff3](/images/maxdiv_ccc_ff3.png)  |  ![maxdiv_ccc_ff3_equity](/images/maxdiv_ccc_ff3_equity.png)

#### Constant Correlation and FF5 factors:
Out-of-sample Diversification |  Equity
:-------------------------:|:-------------------------:
![maxdiv_ccc_ff5](/images/maxdiv_ccc_ff5.png)  |  ![maxdiv_ccc_ff5_equity](/images/maxdiv_ccc_ff5_equity.png)

#### Dynamic Correlation and FF3 factors:
Out-of-sample Diversification |  Equity
:-------------------------:|:-------------------------:
![maxdiv_dcc_ff3](/images/maxdiv_dcc_ff3.png)  |  ![maxdiv_dcc_ff3_equity](/images/maxdiv_dcc_ff3_equity.png)

#### Dynamic Correlation and FF5 factors:
Out-of-sample Diversification |  Equity
:-------------------------:|:-------------------------:
![maxdiv_dcc_ff5](/images/maxdiv_dcc_ff5.png)  |  ![maxdiv_dcc_ff5_equity](/images/maxdiv_dcc_ff5_equity.png)

### Equal Risk Contribution Portfolio:
#### Constant Correlation and FF3 factors:
Out-of-sample Risk Contribution |  Equity
:-------------------------:|:-------------------------:
![rp_ccc_ff3](/images/rp_ccc_ff3.png)  |  ![rp_ccc_ff3_equity](/images/rp_ccc_ff3_equity.png)

#### Constant Correlation and FF5 factors:
Out-of-sample Risk Contribution |  Equity
:-------------------------:|:-------------------------:
![rp_ccc_ff5](/images/rp_ccc_ff5.png)  |  ![rp_ccc_ff5_equity](/images/rp_ccc_ff5_equity.png)

#### Dynamic Correlation and FF3 factors:
Out-of-sample Risk Contribution|  Equity
:-------------------------:|:-------------------------:
![rp_dcc_ff3](/images/rp_dcc_ff3.png)  |  ![rp_dcc_ff3_equity](/images/rp_dcc_ff3_equity.png)

#### Dynamic Correlation and FF5 factors:
Out-of-sample Risk Contribution |  Equity
:-------------------------:|:-------------------------:
![rp_dcc_ff5](/images/rp_dcc_ff5.png)  |  ![rp_dcc_ff5_equity](/images/rp_dcc_ff5_equity.png)
