s = input("Enter Sentence:")
d=l=0
for c in s:
    if c.isdigit():
        d=d+1
    elif c.isalpha():
        l=l+1
    else:
        pass
print("Number of Letters: ", l)
print("Number of Digits:", d)