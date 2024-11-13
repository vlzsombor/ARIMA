


In some use cases, the output should not be continuous (such my apartment has X_1 many floors and X_2 square meter -> I can expect Y money if I sell it) but discrete (The size of the tumor is X_1 is it malignant tumor?)

For such a use case, logistic regression comes in handy.

It can use the sigmoid function to decide if the output is true or false. (A common solution to say, if Y is greater than 0.5 than it is true)

$latex \sigma(z) = \frac{1} {1 + e^{-z}}$ 

If z is small, 1/big -> 0, if z is big, 1/(1+0) -> 1



