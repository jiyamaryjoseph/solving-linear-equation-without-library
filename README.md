# normal-equation-linear-regression

 Here I am using normal equation for predicting data with two different methods..(Matrix Multiplication :) )
 
 

  Linear regression is a method for modeling the relationship between two scalar values: the input variable x and the output variable y.
  
  The model assumes that y is a linear function or a weighted sum of the input variable.
  
                      y = f(x)
Or, stated with the coefficients.

                      y = b0 + b1 . x1
The model can also be used to model an output variable given multiple input variables called multivariate linear regression (below, brackets were added for readability).

                      y = b0 + (b1 . x1) + (b2 . x2) + ...
The objective of creating a linear regression model is to find the values for the coefficient values (b) that minimize the error in the prediction of the output variable y.


                            x11, x12, x13
                       X = (x21, x22, x23)
                            x31, x32, x33
                            x41, x42, x43

                            b1
                       b = (b2)
                            b3

                            y1
                       y = (y2)
                            y3
                            y4
Reformulated, the problem becomes a system of linear equations where the b vector values are unknown. This type of system is referred to as overdetermined because there are more equations than there are unknowns, i.e. each coefficient is used on each row of data.


It is a challenging problem to solve analytically because there are multiple inconsistent solutions, e.g. multiple possible values for the coefficients. Further, all solutions will have some error because there is no line that will pass nearly through all points, therefore the approach to solving the equations must be able to handle that.


##### The way this is typically achieved is by finding a solution where the values for b in the model minimize the squared error. This is called linear least squares.

                      ||X . b - y||^2 = sum i=1 to m ( sum j=1 to n Xij . bj - yi)^2
             
             
             
##### In matrix notation, this problem is formulated using the so-named normal equation:

                        X^T . X . b = X^T . y
This can be re-arranged in order to specify the solution for b as:

                         b = (X^T . X)^-1 . X^T . y
             
             
This can be solved directly, although given the presence of the matrix inverse can be numerically challenging or unstable.

               # solve directly
                 from numpy import array
                 from numpy.linalg import inv
                 from matplotlib import pyplot
                 data = array([
                 [0.05, 0.12],
                 [0.18, 0.22],
                 [0.31, 0.35],
                 [0.42, 0.38],
                 [0.5, 0.49],
                 ])
                 X, y = data[:,0], data[:,1]
                 X = X.reshape((len(X), 1))
                 # linear least squares
                 b = inv(X.T.dot(X)).dot(X.T).dot(y)
                 print(b)
                 # predict using coefficients
                 yhat = X.dot(b)
                 # plot data and predictions
                 pyplot.scatter(X, y)
                 pyplot.plot(X, yhat, color='red')
                 pyplot.show()
