OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
t q[1];
x q[3];
y q[2];
x q[3];
h q[3];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
x q[4];
u3(0, 0, 1.5707963267948966) q[4];
h q[3];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
z q[2];
u3(0, 0, 1.5707963267948966) q[3];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
tdg q[1];
z q[0];
t q[1];
x q[2];
x q[1];
s q[2];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
y q[0];
x q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
z q[4];
x q[2];
x q[4];
sdg q[3];
z q[4];
z q[4];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
