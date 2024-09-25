# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv", sep = ';')

x = df.araba_max_hiz.values.reshape(-1,1)
y = df.araba_fiyat.values.reshape(-1,1)


plt.scatter(x,y)
plt.ylabel("araba fiyat (*1000)")
plt.xlabel("araba fmax hiz")
plt.show()

#linear regression = y = b0 + b1*x
#multiple linear regression = y = b0 + b1*x +b2*x ...

#%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)


#%% PREDICT
y_head = lr.predict(x)

plt.plot(x,y_head,color = "red", label = "linear")
plt.show()


#%% polynomial linear regression y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures( degree = 10 )

x_polynomial = polynomial_regression.fit_transform(x)


#%% Fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial, y)

#%%

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2, color = "black", label = "poly")
plt.legend()
plt.show()






# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("polynomial regression.csv",sep = ";")

# y = df.araba_max_hiz.values.reshape(-1,1)
# x = df.araba_fiyat.values.reshape(-1,1)

# plt.scatter(x,y)
# plt.ylabel("araba_max_hiz")
# plt.xlabel("araba_fiyat")
# plt.show()

# # linear regression =  y = b0 + b1*x
# # multiple linear regression   y = b0 + b1*x1 + b2*x2

# # %% linear regression
# from sklearn.linear_model import LinearRegression

# lr = LinearRegression()

# lr.fit(x,y)

# #%% predict
# y_head = lr.predict(x)

# plt.plot(x,y_head,color="red",label ="linear")
# plt.show()

# print("10 milyon tl lik araba hizi tahmini: ",lr.predict(10000))


# # %%
# # polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

# from sklearn.preprocessing import PolynomialFeatures
# polynomial_regression = PolynomialFeatures(degree = 2)

# x_polynomial = polynomial_regression.fit_transform(x)


# # %% fit
# linear_regression2 = LinearRegression()
# linear_regression2.fit(x_polynomial,y)

# # %%

# y_head2 = linear_regression2.predict(x_polynomial)

# plt.plot(x,y_head2,color= "green",label = "poly")
# plt.legend()
# plt.show()





















