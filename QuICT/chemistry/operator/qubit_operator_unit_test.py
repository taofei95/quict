from qubit_operator import QubitOperator

f_a = QubitOperator("X1 Y4 Z13 X2 Y1",-1.2)
print(f_a)
f_b = QubitOperator("Y2 Y3 X5 Z2",2)
print(f_b)
print(f_a+f_b)

f_a = QubitOperator([(2,1),(8,3),(1,1),(2,2)],8.2)
f_b = QubitOperator([(2,3),(8,3),(1,1)],-0.03)
print(f_a)
print(f_b)
f_c = f_a + f_b
print(f_c)

f_c *= f_a
print(f_c)
f_c -= f_b
print(f_c)
f_c /= 4
print(f_c)
print(f_c.parse())

s=input()
f_s=QubitOperator(s)
f_s /= 2
print(f_s.parse())