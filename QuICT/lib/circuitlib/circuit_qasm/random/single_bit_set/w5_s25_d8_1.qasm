OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
u1(1.5707963267948966) q[0];
s q[4];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[3];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[0];
u3(0, 0, 1.5707963267948966) q[0];
x q[3];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
tdg q[2];
z q[0];
z q[0];
h q[0];
sdg q[0];
t q[3];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[1];
y q[4];
s q[4];
x q[3];
t q[3];
y q[3];
