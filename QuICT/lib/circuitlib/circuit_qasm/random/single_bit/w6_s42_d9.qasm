OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
u3(0, 0, 1.5707963267948966) q[2];
x q[1];
h q[1];
z q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
x q[3];
tdg q[3];
ry(1.5707963267948966) q[2];
x q[2];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
y q[1];
z q[5];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
y q[2];
y q[3];
s q[2];
t q[0];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
y q[4];
t q[5];
tdg q[1];
ry(1.5707963267948966) q[2];
z q[5];
s q[0];
z q[0];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
x q[4];
t q[0];
sdg q[4];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
y q[3];
tdg q[1];