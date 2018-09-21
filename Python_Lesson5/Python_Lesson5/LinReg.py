import numpy as np
import matplotlib.pyplot as plt         #for plotting the given points
x=np.array([2.9,6.7,4.9,7.9,9.8,6.9,6.1,6.2,6,5.1,4.7,4.4,5.8])       #converts the given list into array
y=np.array([4,7.4,5,7.2,7.9,6.1,6,5.8,5.2,4.2,4,4.4,5.2])

meanx=np.mean(x)                        #the meanvalue of x will be stored in meanx
meany=np.mean(y)                          #the meanvalue of y will be stored in meany
num=np.sum((x-meanx)*(y-meany))             #calculate the difference between the mean and given value
den=np.sum(pow((x-meanx),2))               #squares the difference between given x and meanx
m=num/den                                #slope calculation
intercept=meany-(m*meanx)
val=(m*x)+intercept                     #gives us the line equation
plt.plot(x,y,"ro")                              #plots the given x,y values
plt.plot(x,val)
plt.show()                             #plots the points on the graph