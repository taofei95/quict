OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
s q[3];
rx(1.5707963267948966) q[2];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
t q[0];
s q[0];
rz(1.5707963267948966) q[2];
y q[2];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
z q[2];
rz(1.5707963267948966) q[2];
h q[0];
h q[1];
ry(1.5707963267948966) q[1];
x q[1];
s q[2];
tdg q[0];
z q[3];
y q[3];
rz(1.5707963267948966) q[3];
z q[0];
x q[3];
t q[2];
h q[0];
x q[0];
s q[0];
h q[3];
ry(1.5707963267948966) q[3];
t q[2];
tdg q[2];
tdg q[1];
x q[1];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
h q[1];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[0];
rz(1.5707963267948966) q[0];
h q[2];
x q[3];
rz(1.5707963267948966) q[0];
