OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
z q[0];
rz(1.5707963267948966) q[4];
z q[2];
y q[2];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[0];
tdg q[2];
x q[3];
y q[4];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
s q[1];
h q[0];
y q[3];
ry(1.5707963267948966) q[1];
t q[4];
x q[0];
h q[4];
tdg q[3];
tdg q[3];
s q[4];
y q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
z q[1];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
x q[4];
y q[0];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[3];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
tdg q[3];
rz(1.5707963267948966) q[1];
h q[3];
z q[3];
u1(1.5707963267948966) q[3];
y q[3];
u1(1.5707963267948966) q[1];
