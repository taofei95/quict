OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
tdg q[2];
y q[1];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[2];
t q[1];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
t q[0];
tdg q[0];
tdg q[0];
t q[1];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
x q[1];
y q[1];
tdg q[3];
z q[1];
rx(1.5707963267948966) q[2];
y q[2];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
