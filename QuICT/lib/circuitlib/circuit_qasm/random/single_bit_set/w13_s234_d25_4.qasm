OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
z q[3];
s q[8];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[8];
ry(1.5707963267948966) q[2];
x q[10];
rz(1.5707963267948966) q[2];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[0];
u3(0, 0, 1.5707963267948966) q[2];
y q[4];
t q[1];
ry(1.5707963267948966) q[3];
s q[12];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
h q[7];
z q[7];
tdg q[0];
t q[9];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[6];
z q[5];
z q[8];
t q[8];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[12];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[10];
ry(1.5707963267948966) q[4];
x q[1];
y q[11];
u3(0, 0, 1.5707963267948966) q[12];
y q[0];
u1(1.5707963267948966) q[9];
t q[0];
h q[12];
h q[9];
tdg q[2];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[5];
z q[12];
z q[11];
rx(1.5707963267948966) q[9];
t q[12];
s q[2];
tdg q[10];
ry(1.5707963267948966) q[11];
u1(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[7];
y q[9];
z q[11];
u1(1.5707963267948966) q[12];
h q[1];
t q[1];
tdg q[3];
rz(1.5707963267948966) q[11];
h q[6];
t q[8];
sdg q[3];
u1(1.5707963267948966) q[8];
t q[3];
x q[3];
tdg q[12];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[12];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[6];
tdg q[5];
z q[6];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
y q[12];
x q[11];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[4];
h q[11];
t q[6];
tdg q[10];
x q[7];
y q[0];
y q[3];
u1(1.5707963267948966) q[7];
t q[5];
u1(1.5707963267948966) q[3];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[8];
x q[1];
z q[8];
x q[9];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[4];
s q[7];
ry(1.5707963267948966) q[0];
y q[7];
sdg q[7];
h q[2];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
y q[9];
rx(1.5707963267948966) q[9];
z q[9];
h q[6];
x q[6];
h q[12];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[3];
sdg q[0];
rx(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[6];
s q[4];
z q[12];
u1(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[12];
s q[2];
rx(1.5707963267948966) q[8];
z q[3];
u3(0, 0, 1.5707963267948966) q[11];
x q[3];
h q[12];
z q[12];
u3(0, 0, 1.5707963267948966) q[12];
rz(1.5707963267948966) q[10];
sdg q[6];
x q[7];
ry(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
sdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[7];
y q[5];
u1(1.5707963267948966) q[6];
t q[2];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[5];
t q[0];
ry(1.5707963267948966) q[11];
y q[10];
s q[0];
y q[3];
u3(0, 0, 1.5707963267948966) q[9];
t q[1];
z q[0];
u3(0, 0, 1.5707963267948966) q[11];
y q[1];
ry(1.5707963267948966) q[5];
u1(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[12];
t q[12];
x q[1];
y q[1];
h q[7];
y q[0];
u1(1.5707963267948966) q[12];
h q[6];
s q[6];
t q[12];
u3(0, 0, 1.5707963267948966) q[12];
sdg q[2];
z q[10];
z q[10];
t q[7];
z q[1];
sdg q[6];
rx(1.5707963267948966) q[11];
tdg q[0];
t q[2];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[10];
t q[0];
s q[11];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
z q[0];
t q[10];
sdg q[5];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[1];
sdg q[5];
x q[3];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
y q[10];
rx(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[10];
u2(1.5707963267948966, 1.5707963267948966) q[5];
y q[6];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[2];
rx(1.5707963267948966) q[12];
x q[12];
u3(0, 0, 1.5707963267948966) q[6];
h q[2];
sdg q[3];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[1];
s q[0];
ry(1.5707963267948966) q[8];
s q[6];
y q[1];
u1(1.5707963267948966) q[2];
h q[2];
x q[2];
x q[2];
y q[7];
sdg q[10];
tdg q[7];
h q[3];
s q[9];
ry(1.5707963267948966) q[10];
h q[2];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[8];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
