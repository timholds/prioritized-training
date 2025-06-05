The paper states (in Section 3.2):

"We train for a fixed number of steps (not epochs) and evaluate the test accuracy at regular intervals."

So, they are comparing the same number of steps (i.e., same number of weight updates) for both methods.

If a susample How do they ensure that the prioritized method does not run out of data?

- They use a "big batch" of candidate points (e.g., 10% of the training set) which is difrefreshed every epoch. The small batch (the actual training batch) is selected from this big batch. When the big batch is exhausted (i.e., after a certain number of steps within the epoch), they refresh the big batch (by randomly sampling a new big batch from the training set). This is done without regard to whether the entire training set has been seen.


They test out two different target accuracies in the paper for each dataset. I'm taking the midpoint between those two values for my tests. 
