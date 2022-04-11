This is where the saved models are for the code.

1. The model.hdf5 was provided by Ribeiro et al. and requires no training
2. back_model_best.hdf5 is just the best model during the last fold in the k-fold algorithm. This will be updated during the k-fold process and was just stored here for consistency
3. RF_params.pickle is for the RF model and holds the best estimator based on the randomized search
