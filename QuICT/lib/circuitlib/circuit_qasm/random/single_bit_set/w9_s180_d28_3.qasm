OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
t q[3];
x q[5];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[1];
z q[0];
rz(1.5707963267948966) q[6];
z q[0];
y q[7];
rx(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[7];
x q[7];
s q[4];
x q[1];
tdg q[8];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[0];
y q[3];
y q[6];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[7];
x q[4];
h q[6];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[1];
ry(1.5707963267948966) q[5];
t q[1];
x q[3];
x q[8];
s q[2];
rz(1.5707963267948966) q[3];
tdg q[0];
t q[4];
u3(0, 0, 1.5707963267948966) q[2];
s q[8];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[0];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
sdg q[2];
y q[8];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
s q[6];
s q[8];
sdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[6];
u1(1.5707963267948966) q[8];
x q[8];
u1(1.5707963267948966) q[0];
s q[5];
s q[6];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
sdg q[4];
sdg q[0];
ry(1.5707963267948966) q[8];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
y q[0];
tdg q[7];
t q[1];
z q[8];
u1(1.5707963267948966) q[5];
tdg q[4];
y q[1];
s q[8];
t q[0];
u3(0, 0, 1.5707963267948966) q[8];
sdg q[1];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[6];
s q[5];
u1(1.5707963267948966) q[3];
sdg q[0];
rz(1.5707963267948966) q[8];
s q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
t q[7];
rz(1.5707963267948966) q[5];
t q[2];
sdg q[8];
y q[2];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
y q[2];
rz(1.5707963267948966) q[7];
t q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[5];
x q[4];
x q[0];
tdg q[8];
tdg q[6];
x q[6];
rz(1.5707963267948966) q[8];
sdg q[4];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[8];
t q[2];
z q[1];
t q[8];
z q[1];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[6];
s q[4];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[4];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[4];
y q[4];
h q[2];
u1(1.5707963267948966) q[4];
h q[6];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[8];
x q[7];
y q[3];
tdg q[3];
ry(1.5707963267948966) q[7];
s q[6];
tdg q[5];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
t q[8];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[6];
x q[4];
u1(1.5707963267948966) q[3];
s q[6];
tdg q[8];
tdg q[1];
z q[4];
ry(1.5707963267948966) q[8];
z q[0];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[3];
t q[0];
rz(1.5707963267948966) q[8];
h q[0];
z q[3];
ry(1.5707963267948966) q[0];
tdg q[1];
tdg q[5];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[8];
u1(1.5707963267948966) q[3];
z q[4];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
