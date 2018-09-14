class Employee():
    employees_count = 0
    total_salary = 0
    avg_salary = 0

    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.employees_count += 1
        Employee.total_salary = Employee.total_salary + salary

    # Function for average salary
    def avg_salary(self):
        avg_sal = float(Employee.total_salary / Employee.employees_count)
        print(f'The avarage salary of total number employees is {avg_sal}')

    # total count of employees
    def count_employee(self):
        print(f'Total number of employees are {Employee.employees_count} ')

    # display the details of the employee
    def show_details(self):
        print(f' name : {self.name} \n family : {self.family} \n salary : {self.salary} \
        \n department : {self.department}')


# inherited class from Employee
class Fulltime_emp(Employee):
    def _init_(self, nme, fmly, slry, dept):
        Employee._init_(self, nme, fmly, slry, dept)



emp1 = Employee("jack","linen",12000,"Web developer")
emp2 = Employee("woxen","lingush",17021,"IOS developer")
emp3 = Employee("nick","martial",1212,"Anroid developer")
emp4 = Employee("sanchez","alexies",12132," Data analyst")
emp5 = Employee("remisty","kingon",145011,"Data scientist")
f_emp = Fulltime_emp("Harika","Harres",12234,"Python_developer")
f_emp.show_details()
emp1.avg_salary()
emp1.count_employee()
emp1.show_details()
emp1.count_employee()
emp1.avg_salary()
Employee.employees_count