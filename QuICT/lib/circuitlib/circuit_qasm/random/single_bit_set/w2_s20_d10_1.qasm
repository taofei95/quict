OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(1.5707963267948966) q[1];
sdg q[0];
h q[1];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
z q[1];
u1(1.5707963267948966) q[0];
t q[1];
x q[0];
tdg q[0];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
tdg q[1];
ry(1.5707963267948966) q[0];
z q[0];
s q[1];
x q[1];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
