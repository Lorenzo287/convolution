import numpy as np

a = (1, 2, 3)
b = (4, 5, 6)

""" 
    flip b, dot product with a (of the overlapping numbers) while sliding 

          1  2  3       1  2  3      1  2  3
    6  5  4          6  5  4         6  5  4
    -------------    ----------      -------
          4             5+ 8 = 13    6+10+12 = 28

    c = (4, 13, 28, ...)
"""

c = np.convolve(a, b)
print(c)
