## Intent Classification
This project uses the facebook intent data (same as used for intents/slots PoC) to train a classifier for predicting top-level intent. The main model is a 2-layer BiLSTM which achieves >90% classification accuracy across the 19 possible categories.
Top-intent categories are not aligned with our ultimate goal of predicting target-entity types, but results show that this task is feasible given a moderate amount of training data.
