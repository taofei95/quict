OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
tdg q[10];
h q[5];
tdg q[6];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[3];
x q[4];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
tdg q[2];
h q[2];
u1(1.5707963267948966) q[7];
s q[5];
tdg q[0];
rz(1.5707963267948966) q[0];
y q[2];
h q[3];
rz(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[9];
h q[3];
y q[0];
rz(1.5707963267948966) q[10];
x q[7];
z q[9];
tdg q[8];
z q[8];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[8];
z q[5];
u1(1.5707963267948966) q[7];
t q[8];
u3(0, 0, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[2];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[10];
t q[4];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
y q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[10];
s q[0];
x q[10];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[0];
s q[1];
s q[4];
z q[3];
u3(0, 0, 1.5707963267948966) q[0];
s q[10];
u1(1.5707963267948966) q[2];
t q[0];
t q[2];
h q[8];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[3];
h q[8];
rz(1.5707963267948966) q[5];
s q[8];
u2(1.5707963267948966, 1.5707963267948966) q[0];
z q[10];
sdg q[5];
ry(1.5707963267948966) q[4];
t q[8];
s q[9];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[4];
