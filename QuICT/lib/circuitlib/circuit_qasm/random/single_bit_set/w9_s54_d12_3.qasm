OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
rz(1.5707963267948966) q[0];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
t q[2];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
s q[4];
s q[3];
t q[4];
sdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[5];
y q[5];
u1(1.5707963267948966) q[4];
x q[2];
sdg q[2];
sdg q[8];
y q[4];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[5];
x q[0];
t q[5];
t q[5];
u3(0, 0, 1.5707963267948966) q[8];
x q[0];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
s q[6];
tdg q[0];
rx(1.5707963267948966) q[2];
tdg q[3];
u1(1.5707963267948966) q[5];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[8];
y q[6];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[2];
z q[4];
t q[2];
z q[1];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[1];
h q[7];
y q[2];
tdg q[2];
z q[3];
h q[7];
rx(1.5707963267948966) q[1];
