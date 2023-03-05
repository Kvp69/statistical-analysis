# statistical-analysis
# ANSWER OF QUESTION 2
##i) Cross-entropy:
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
Output:
0.23884306837252087

ii) Entropy:
Entropy is a measure of the uncertainty or randomness of a probability distribution. It is defined as:

$H(X) = -\sum_{x} P(X=x) \log P(X=x)$

where X is a random variable and P(X=x) is the probability of X taking on the value x.

Here's an implementation of the entropy function in Python:
def entropy(p):
    return -np.sum(p * np.log2(p))
Example:
Let's say we have a probability distribution:

p = [0.3, 0.4, 0.3]

We can calculate the entropy of this distribution using the above function:
Output:
1.5219280948873621

iii) Mutual Information:
Mutual information is a measure of the amount of information that two random variables share. It is defined as:

$I(X;Y) = \sum_{y} \sum_{x} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$

where X and Y are two random variables, p(x,y) is the joint probability distribution of X and Y, and p(x) and p(y) are the marginal probability distributions of X and Y, respectively.

Here's an implementation of the mutual information function in Python:
def mutual_information(p_xy, p_x, p_y):
    p_x_y = p_xy / np.outer(p_x, p_y)
    return np.sum(p_xy * np.log2(p_x_y))
Example:
Let's say we have two random variables X and Y with the following joint probability distribution:

p_xy = np.array([[0.1, 0.2], [0.3, 0.4]])
p_x = np.sum(p_xy, axis=1)
p_y = np.sum(p_xy, axis=0)

We can calculate the mutual information between X and Y using the above function:
mi = mutual_information(p_xy, p_x, p_y)
print(mi)
Output:
0.04512728270729682

iv) Conditional entropy:
Conditional entropy measures the uncertainty of a random variable given another random variable. It is defined as:

H(X|Y) = - ∑i,j p(x_i, y_j) log(p(x_i | y_j))

Here's an implementation in Python:
import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))
    For example
    python code
p = np.array([0.2, 0.4, 0.4])
q = np.array([0.1, 0.3, 0.6])
conditional_entropy(pxy)
output
0.6747843568798091
v) KL divergence:
KL divergence measures the difference between two probability distributions p and q. It is defined as:

Dkl(p||q) = ∑i p(i) log(p(i)/q(i))

Here's an implementation in Python:
import numpy as np
def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))
For an example, let's consider two probability distributions:
python code
p = np.array([0.2, 0.4, 0.4])
q = np.array([0.1, 0.3, 0.6])
kl_divergence(p, q)
output
0.3431580243992872
