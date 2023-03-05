# statistical-analysis
# ANSWER OF QUESTION 2
#i) Cross-entropy:
    Cross-entropy is a measure of the dissimilarity between two probability distributions. It is often used in machine learning as a loss function for classification         problems. The formula for cross-entropy is as follows:
    H(p, q) = -\sum_{x} p(x) \log q(x)
    where p and q are two probability distributions over the same set of events x.
    Here's an implementation of the cross-entropy function in Python:
    import numpy as np
       def cross_entropy(p, q):
       return -np.sum(p * np.log(q))
   Example:
   Let's say we have two probability distributions:
  p = [0.2, 0.3, 0.5]
  q = [0.3, 0.2, 0.5]

We can calculate the cross-entropy between them using the above function:
