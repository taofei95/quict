OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
sdg q[0];
s q[1];
x q[1];
y q[0];
x q[1];
sdg q[1];
t q[1];
h q[1];
x q[0];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[0];
y q[1];
tdg q[1];
ry(1.5707963267948966) q[0];
t q[1];
s q[0];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
z q[1];
z q[1];
rx(1.5707963267948966) q[1];
z q[0];
y q[1];
z q[1];
tdg q[1];
u1(1.5707963267948966) q[0];
z q[0];
y q[1];
rx(1.5707963267948966) q[0];