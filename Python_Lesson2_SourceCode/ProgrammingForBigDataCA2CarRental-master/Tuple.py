num_list = []
for i in range(6):
    num_list.append(int(input("enter a number")))
n = len(num_list)
print(f'number list is {num_list}')
num_tuple = (num_list[0],num_list[n-1])
print(num_tuple)