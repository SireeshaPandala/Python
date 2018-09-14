import numpy as np
# Initialising the random vector of size 15 in range 0 to 20 using "numpy"
random_vec =np.random.randint(0,20, size=15)
print(f'The random vector is :  {random_vec}')
# finding the most similar values in the vector
count_similar_items = np.bincount(random_vec)
freq_num = np.argmax(count_similar_items)
print(f'the most frequent item in the vector is {freq_num}')