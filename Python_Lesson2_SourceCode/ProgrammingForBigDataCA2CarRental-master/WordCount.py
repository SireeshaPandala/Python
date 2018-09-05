fo = open("text","r")
my_string1 = fo.read()
print(str)

#my_string1 = input("Enter the string 1: ")
# my_string2 = input("Enter the string 2: ")
list1 = my_string1.split()
# list2 = my_string2.split()
character_count = 0
word_count = 0
for item in list1:
    character_count += len(item)
    word_count += 1

print(f' for {my_string1} character_count is  {character_count} and word count is {word_count}')

fo.close()