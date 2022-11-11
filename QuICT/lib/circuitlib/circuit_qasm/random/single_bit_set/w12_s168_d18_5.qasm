OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
rz(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[5];
z q[5];
z q[10];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[0];
s q[1];
z q[3];
rz(1.5707963267948966) q[5];
t q[7];
z q[10];
y q[1];
rz(1.5707963267948966) q[0];
s q[3];
ry(1.5707963267948966) q[4];
x q[6];
u3(0, 0, 1.5707963267948966) q[11];
t q[3];
tdg q[5];
rx(1.5707963267948966) q[1];
z q[9];
u1(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[3];
t q[6];
rz(1.5707963267948966) q[11];
z q[2];
u3(0, 0, 1.5707963267948966) q[0];
h q[1];
tdg q[4];
rz(1.5707963267948966) q[4];
x q[11];
h q[9];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[11];
u1(1.5707963267948966) q[4];
tdg q[9];
sdg q[10];
sdg q[8];
ry(1.5707963267948966) q[11];
h q[11];
y q[0];
u3(0, 0, 1.5707963267948966) q[8];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[0];
y q[6];
z q[3];
tdg q[7];
s q[7];
s q[9];
tdg q[9];
s q[1];
z q[9];
s q[7];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
z q[2];
s q[11];
u3(0, 0, 1.5707963267948966) q[11];
y q[5];
s q[1];
t q[11];
h q[9];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
sdg q[2];
u1(1.5707963267948966) q[3];
x q[2];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[2];
z q[8];
h q[1];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[9];
z q[1];
s q[1];
u1(1.5707963267948966) q[8];
h q[7];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
u1(1.5707963267948966) q[5];
t q[2];
u1(1.5707963267948966) q[0];
z q[3];
ry(1.5707963267948966) q[3];
z q[10];
ry(1.5707963267948966) q[3];
t q[9];
s q[6];
rz(1.5707963267948966) q[4];
h q[7];
x q[9];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
tdg q[5];
rx(1.5707963267948966) q[4];
x q[5];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
u1(1.5707963267948966) q[0];
s q[1];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[7];
y q[0];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[6];
z q[3];
ry(1.5707963267948966) q[10];
u1(1.5707963267948966) q[3];
y q[7];
x q[8];
sdg q[0];
h q[3];
u3(0, 0, 1.5707963267948966) q[2];
h q[5];
rx(1.5707963267948966) q[4];
h q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[8];
s q[9];
tdg q[3];
s q[11];
rz(1.5707963267948966) q[4];
x q[11];
u3(0, 0, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[9];
z q[10];
t q[11];
u3(0, 0, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[8];
x q[6];
t q[4];
h q[1];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
z q[8];
sdg q[11];
x q[11];
x q[4];
sdg q[10];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
tdg q[9];
s q[6];
y q[2];
z q[10];
ry(1.5707963267948966) q[2];
tdg q[3];
rz(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[10];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[4];
y q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[0];
x q[7];
ry(1.5707963267948966) q[2];
