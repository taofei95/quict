from fermion_operator import FermionOperator

f_a = FermionOperator("1^ 4^ 13 2 405 2^ 5^ 5 ",-1.2)
print(f_a)
f_b = FermionOperator("  1^ 2^ 1 1^ 1",2)
print(f_b)
print(f_a+f_b)

f_a = FermionOperator([(2,1),(8,0),(1,1),(2,0)],4008.2)
f_b = FermionOperator([(2,0),(8,0),(1,1),(2,1)],-0.03)
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
f_s=FermionOperator(s)
f_s /= 2
print(f_s.parse())