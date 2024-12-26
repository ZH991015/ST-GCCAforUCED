# ST-GCCAforUCED
The ST-GCCA.py in the model file is the code of the model used in this paper ‘Space Tree-based Graph Continuous Cellular Automaton for Unit Commitment and Economic Dispatch Optimisation’.
ST-GCCA.py in the model file is the code of the model used in this paper ‘Space Tree-based Graph Continuous Cellular Automaton for Unit Commitment and Economic Dispatch Optimization’. data in the data file is the reference data, and the source is the output of the ST-GCCA model.

data：The dataset used to validate the results and plot them is included in this document, which contains the power scheduling plans of 54 generators for 2880 moments (10 days), where Pred is the predicted value of the ST model, Adjust is the output value of the ST-GCCA, and Actual is the real label.

ST-GCCAmodel：The input to the model is the load demand while the output is the generator scheduling plan

Note: The data in the data file cannot be directly used to run the ST-GCCA model. It is a fragment extracted from the ST-GCCA output, used for final effect verification. Some constraints in the Pred section cannot be satisfied. It is worth mentioning that the dataset provided by the State Grid specified incorrect initial power for generating units, which results in some constraints in the Actual label that cannot be satisfied. However, the Adjust output from ST-GCCA can fully meet all constraints.
