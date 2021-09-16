### Inverted Pendulum example

* To generate the data using sensitivity-based data augmentation technique, run main_GenTrainData.m. This generates the sparsely sampled data set, full data set where the additional data samples are generated exactly by solving the full NLP, and the augmented data set generated using the tangential predictor.

* main_CL_sim.m - runs the closed loop simulation either using the exact MPC or the approximate MPC policy trained using the three data sets.  

Code requires CasADi v3.5.1

This is a working document and is subject to change. 
