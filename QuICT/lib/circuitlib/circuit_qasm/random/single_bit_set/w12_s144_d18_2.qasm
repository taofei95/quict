OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[7];
sdg q[5];
u1(1.5707963267948966) q[7];
x q[8];
tdg q[4];
h q[2];
rz(1.5707963267948966) q[11];
x q[9];
tdg q[6];
z q[2];
y q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
s q[5];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[2];
z q[4];
u3(0, 0, 1.5707963267948966) q[11];
x q[9];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[5];
rz(1.5707963267948966) q[9];
s q[1];
t q[10];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
x q[0];
x q[11];
z q[0];
rx(1.5707963267948966) q[3];
z q[1];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
y q[0];
y q[10];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[2];
x q[11];
y q[4];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[0];
h q[3];
sdg q[2];
z q[4];
x q[4];
ry(1.5707963267948966) q[9];
y q[7];
t q[3];
s q[1];
tdg q[3];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
z q[9];
z q[7];
t q[8];
z q[9];
x q[0];
h q[8];
sdg q[5];
h q[3];
sdg q[1];
rx(1.5707963267948966) q[3];
s q[0];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
y q[3];
t q[7];
x q[7];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[9];
t q[3];
u3(0, 0, 1.5707963267948966) q[8];
z q[7];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[9];
rz(1.5707963267948966) q[11];
s q[4];
sdg q[7];
ry(1.5707963267948966) q[11];
s q[7];
x q[0];
t q[0];
y q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
y q[10];
h q[3];
tdg q[11];
t q[4];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[3];
s q[11];
t q[5];
y q[2];
u2(1.5707963267948966, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[6];
s q[6];
rx(1.5707963267948966) q[7];
y q[4];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[11];
u1(1.5707963267948966) q[11];
s q[6];
t q[1];
ry(1.5707963267948966) q[11];
u1(1.5707963267948966) q[3];
x q[0];
rx(1.5707963267948966) q[4];
sdg q[8];
tdg q[11];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[11];
y q[11];
rx(1.5707963267948966) q[9];
h q[11];
s q[1];
u3(0, 0, 1.5707963267948966) q[5];
z q[0];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[11];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
u1(1.5707963267948966) q[9];
x q[10];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[6];
tdg q[11];
x q[4];
rz(1.5707963267948966) q[6];
