OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rx(1.5707963267948966) q[0];
s q[1];
ry(1.5707963267948966) q[4];
y q[3];
t q[1];
sdg q[8];
x q[6];
rx(1.5707963267948966) q[9];
x q[5];
y q[3];
tdg q[0];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
x q[5];
t q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
h q[1];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[9];
x q[9];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[9];
x q[4];
rz(1.5707963267948966) q[0];
h q[1];
x q[8];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
z q[8];
h q[1];
s q[2];
u1(1.5707963267948966) q[2];
y q[2];
t q[5];
x q[3];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[8];
sdg q[4];
x q[6];
rz(1.5707963267948966) q[9];
tdg q[2];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[9];
z q[2];
sdg q[0];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
tdg q[6];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[1];
y q[5];
x q[4];
y q[6];
z q[5];
sdg q[2];
u1(1.5707963267948966) q[9];
y q[7];
rx(1.5707963267948966) q[7];
z q[6];
tdg q[8];
x q[2];
sdg q[5];
t q[5];
h q[9];
u1(1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[7];
