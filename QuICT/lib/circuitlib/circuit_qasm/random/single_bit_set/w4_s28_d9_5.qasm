OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
s q[2];
y q[3];
s q[1];
y q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
tdg q[3];
s q[3];
y q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
t q[2];
y q[2];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[0];
rx(1.5707963267948966) q[0];
x q[2];
sdg q[2];
tdg q[0];
tdg q[0];
z q[0];
rx(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
t q[0];
x q[2];
rz(1.5707963267948966) q[1];
