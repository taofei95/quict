OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
z q[3];
tdg q[0];
z q[3];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[1];
y q[2];
y q[3];
t q[2];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[3];
x q[0];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
sdg q[2];
h q[3];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[3];
y q[2];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
z q[2];
h q[2];
z q[1];
rx(1.5707963267948966) q[1];
s q[3];
tdg q[2];
u1(1.5707963267948966) q[2];
