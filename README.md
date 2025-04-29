# quad-quad

Fig 1:

keras_models.py contains functions to build the quadratic-quadratic model and variants. spearmint_runner.py is used by Spearmint package (rjrowekamp/Spearmint) to fit models and find optimal hyperparameters. config_??.json are used by Spearmint to define the hyperparameters to be fit for each variant of the model. The two letters indicate whether the model variant is linear (L) or quadratic (Q) in the respective layer.

corr_tools.py contains functions to calculated extrapolated correlations from repeated test stimuli and responses as well as to fit random forest models using xgboost (MLEncoding) as a performance benchmark.

Fig 2:

layer_stats.py contains functions to calculate the variances contributed by the linear or quadratic features of each layer of the model and calculate a quadratic index that measures their relative dominance.

Fig 3:

diff_evol.py has a framework to run differential evolution. diff_evol_func.py has functions to be optimized by that framework, includnig the curved Gabors used by this figgure. angle_tools.py has functions to calculate the local orientation preferences of the Gabors fit to the excitatory and suppressive components of the quadratic feature and measure how they compare to each other.

Fig 4:

angle_tools.py has functions to measure the direction selectivity of quadratic features using FFT.

Fig 5:

angle_tools.py is used again to compare features from different layers.

Fig 6:

sparse_tools.py has functions to measure the sparseness of a model's response as well as function to rotate portions of the model's parameters relative to each other.

S1 Fig:

misc_tools.py has functions to calculate the AIC for different models.

S2 Fig:

angle_tools.py has functions to measure motion selectivity.

S3 Fig:

angle_tools.py has a function to measure the orientation distribution in an array, such as stimuli or the outputs of intermediate layers.

S4 Fig:

misc_tools.py has a function to estimate the number of signficant dimensions in a quadratic feature

S5 Fig:

The numbers of significant features from S4 Fig were used to measure their relative contribution by both number and weight.

S6-S9 Figs:

Fig 7 reproduced as scatterplots instead of bars.

S10 Fig:

angle_tools.py has a function to measure stimulus statistics for different spatiotemporal frequencies. motion_test.py creates a motion selective model that can be used to test the effect of the different stimulus statistics on the ability to detect motion selectivity.

S11 Fig:

The gratings that maximize and minimize the response of the model fit to a neuron shown in Fig 4. They were fit by a function in diff_evol_func.py.

S12 Fig:

Fig 2 reproduced using models fit with a RELU instead of a sigmoid function in the first layer. These models were fit using spearmint_runner_relu.py

S13 Fig:

Cicular variance of model responses to sine gratings to test magnitude of direction selectivity.
