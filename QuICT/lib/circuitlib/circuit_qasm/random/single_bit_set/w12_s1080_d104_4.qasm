OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[10];
s q[9];
tdg q[6];
s q[8];
t q[9];
rz(1.5707963267948966) q[4];
s q[1];
t q[7];
h q[1];
rz(1.5707963267948966) q[3];
z q[3];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[0];
tdg q[2];
s q[10];
h q[6];
y q[3];
z q[9];
u3(0, 0, 1.5707963267948966) q[6];
tdg q[11];
sdg q[11];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[11];
tdg q[8];
sdg q[2];
x q[7];
x q[7];
x q[0];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[8];
x q[10];
h q[7];
sdg q[0];
h q[10];
t q[6];
sdg q[5];
y q[3];
y q[3];
s q[0];
ry(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[4];
rz(1.5707963267948966) q[11];
s q[11];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[9];
sdg q[4];
z q[8];
t q[1];
h q[1];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[7];
ry(1.5707963267948966) q[8];
sdg q[10];
y q[10];
sdg q[10];
u1(1.5707963267948966) q[4];
s q[0];
rx(1.5707963267948966) q[4];
z q[2];
y q[6];
t q[0];
sdg q[9];
rx(1.5707963267948966) q[5];
tdg q[10];
y q[8];
y q[3];
z q[7];
t q[6];
u1(1.5707963267948966) q[5];
h q[8];
z q[10];
s q[0];
s q[2];
s q[8];
t q[9];
z q[2];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rx(1.5707963267948966) q[0];
t q[4];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
h q[9];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[7];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[6];
x q[4];
t q[7];
z q[2];
s q[8];
sdg q[5];
t q[10];
u3(0, 0, 1.5707963267948966) q[0];
z q[0];
tdg q[0];
h q[2];
y q[4];
rz(1.5707963267948966) q[0];
tdg q[7];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[8];
h q[8];
u3(0, 0, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
x q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[10];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[1];
t q[10];
rz(1.5707963267948966) q[1];
x q[3];
s q[5];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[10];
t q[1];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[4];
s q[2];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[6];
s q[3];
tdg q[3];
ry(1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
tdg q[8];
y q[8];
z q[2];
s q[4];
rz(1.5707963267948966) q[5];
y q[11];
h q[6];
u3(0, 0, 1.5707963267948966) q[11];
tdg q[6];
sdg q[8];
h q[11];
u1(1.5707963267948966) q[0];
h q[1];
x q[2];
s q[2];
u3(0, 0, 1.5707963267948966) q[5];
x q[7];
y q[8];
h q[8];
x q[8];
t q[7];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[11];
s q[1];
tdg q[8];
h q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
x q[10];
s q[2];
y q[11];
z q[8];
u3(0, 0, 1.5707963267948966) q[4];
x q[6];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[8];
z q[1];
rx(1.5707963267948966) q[1];
s q[9];
rz(1.5707963267948966) q[10];
s q[4];
y q[2];
x q[3];
z q[10];
z q[1];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
h q[9];
y q[0];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[9];
x q[8];
tdg q[6];
t q[0];
tdg q[10];
ry(1.5707963267948966) q[6];
t q[3];
rz(1.5707963267948966) q[6];
h q[6];
ry(1.5707963267948966) q[2];
y q[5];
s q[2];
rz(1.5707963267948966) q[6];
tdg q[11];
y q[11];
u1(1.5707963267948966) q[2];
sdg q[4];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[10];
z q[11];
x q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
x q[0];
rz(1.5707963267948966) q[11];
tdg q[2];
h q[6];
tdg q[11];
rz(1.5707963267948966) q[11];
z q[5];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[11];
rz(1.5707963267948966) q[2];
sdg q[9];
ry(1.5707963267948966) q[8];
z q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[11];
s q[4];
sdg q[5];
tdg q[0];
t q[11];
ry(1.5707963267948966) q[2];
h q[11];
t q[11];
u1(1.5707963267948966) q[3];
sdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
y q[4];
x q[4];
sdg q[3];
x q[3];
z q[11];
x q[8];
sdg q[5];
rx(1.5707963267948966) q[9];
h q[10];
u2(1.5707963267948966, 1.5707963267948966) q[5];
y q[7];
u2(1.5707963267948966, 1.5707963267948966) q[8];
z q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[7];
rx(1.5707963267948966) q[10];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[1];
t q[5];
z q[1];
rx(1.5707963267948966) q[8];
t q[4];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[10];
t q[4];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
tdg q[10];
s q[2];
u3(0, 0, 1.5707963267948966) q[0];
t q[3];
u1(1.5707963267948966) q[4];
z q[2];
t q[9];
rx(1.5707963267948966) q[9];
s q[11];
z q[2];
u3(0, 0, 1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[8];
sdg q[5];
x q[8];
h q[7];
y q[7];
s q[5];
u3(0, 0, 1.5707963267948966) q[8];
h q[8];
h q[6];
rz(1.5707963267948966) q[11];
z q[4];
u1(1.5707963267948966) q[10];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[9];
x q[3];
z q[4];
ry(1.5707963267948966) q[11];
y q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
sdg q[1];
s q[7];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[2];
z q[6];
tdg q[6];
z q[10];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
s q[9];
ry(1.5707963267948966) q[6];
s q[7];
u1(1.5707963267948966) q[0];
y q[11];
u3(0, 0, 1.5707963267948966) q[6];
y q[2];
z q[4];
x q[2];
u2(1.5707963267948966, 1.5707963267948966) q[7];
x q[3];
tdg q[2];
y q[7];
u2(1.5707963267948966, 1.5707963267948966) q[7];
tdg q[1];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
z q[0];
u1(1.5707963267948966) q[6];
sdg q[8];
x q[4];
sdg q[6];
t q[11];
u1(1.5707963267948966) q[5];
s q[7];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
y q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
t q[9];
y q[11];
sdg q[11];
ry(1.5707963267948966) q[10];
z q[9];
x q[8];
z q[7];
tdg q[5];
h q[9];
t q[3];
x q[9];
t q[1];
z q[0];
z q[3];
y q[4];
u3(0, 0, 1.5707963267948966) q[9];
x q[4];
ry(1.5707963267948966) q[10];
t q[11];
ry(1.5707963267948966) q[0];
tdg q[8];
h q[8];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[9];
z q[10];
h q[2];
t q[9];
rx(1.5707963267948966) q[1];
z q[10];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[10];
y q[8];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[7];
h q[1];
u1(1.5707963267948966) q[8];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[0];
h q[7];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[5];
t q[9];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
t q[8];
sdg q[3];
h q[5];
z q[10];
x q[3];
sdg q[1];
rx(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[6];
y q[4];
h q[3];
h q[7];
u2(1.5707963267948966, 1.5707963267948966) q[7];
x q[8];
u3(0, 0, 1.5707963267948966) q[0];
z q[3];
s q[6];
t q[11];
sdg q[6];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[10];
h q[8];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[9];
x q[10];
sdg q[2];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[10];
t q[11];
h q[6];
x q[6];
z q[7];
u1(1.5707963267948966) q[1];
sdg q[4];
t q[5];
t q[11];
h q[1];
rx(1.5707963267948966) q[6];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[9];
t q[10];
rz(1.5707963267948966) q[1];
h q[6];
s q[9];
u1(1.5707963267948966) q[7];
s q[6];
z q[0];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[7];
tdg q[10];
tdg q[11];
u1(1.5707963267948966) q[4];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[7];
y q[1];
y q[3];
z q[6];
tdg q[2];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[1];
z q[2];
y q[3];
tdg q[11];
rx(1.5707963267948966) q[4];
sdg q[7];
rx(1.5707963267948966) q[4];
y q[9];
tdg q[8];
sdg q[10];
t q[4];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
h q[8];
z q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[11];
u2(1.5707963267948966, 1.5707963267948966) q[3];
y q[1];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[3];
t q[10];
sdg q[4];
tdg q[11];
y q[0];
y q[1];
u2(1.5707963267948966, 1.5707963267948966) q[10];
s q[7];
x q[3];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
z q[2];
y q[8];
s q[5];
t q[0];
u1(1.5707963267948966) q[8];
s q[6];
h q[10];
h q[0];
rz(1.5707963267948966) q[10];
y q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
x q[9];
u1(1.5707963267948966) q[2];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[11];
sdg q[11];
u2(1.5707963267948966, 1.5707963267948966) q[11];
tdg q[0];
rz(1.5707963267948966) q[4];
t q[4];
t q[4];
ry(1.5707963267948966) q[7];
x q[11];
tdg q[8];
z q[0];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[5];
tdg q[9];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
h q[2];
rx(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[11];
z q[4];
s q[8];
tdg q[1];
rx(1.5707963267948966) q[2];
z q[7];
y q[3];
sdg q[4];
s q[1];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[8];
x q[7];
s q[6];
tdg q[3];
tdg q[3];
rz(1.5707963267948966) q[2];
h q[7];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[0];
h q[0];
s q[11];
u3(0, 0, 1.5707963267948966) q[4];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[8];
x q[8];
z q[0];
s q[7];
u3(0, 0, 1.5707963267948966) q[4];
x q[8];
rz(1.5707963267948966) q[2];
y q[0];
u1(1.5707963267948966) q[9];
s q[8];
z q[9];
u3(0, 0, 1.5707963267948966) q[3];
y q[7];
rz(1.5707963267948966) q[2];
s q[6];
x q[4];
y q[4];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
x q[0];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[6];
sdg q[8];
sdg q[4];
y q[3];
h q[4];
t q[1];
tdg q[2];
x q[2];
rz(1.5707963267948966) q[5];
y q[5];
u3(0, 0, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[6];
t q[7];
sdg q[9];
ry(1.5707963267948966) q[1];
h q[6];
rz(1.5707963267948966) q[0];
t q[2];
y q[1];
z q[9];
u1(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[10];
h q[6];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[3];
z q[11];
z q[7];
s q[8];
u3(0, 0, 1.5707963267948966) q[0];
z q[9];
t q[10];
s q[1];
sdg q[8];
h q[11];
y q[5];
z q[9];
t q[5];
u3(0, 0, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
sdg q[9];
u1(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[10];
z q[5];
x q[7];
sdg q[5];
y q[5];
u3(0, 0, 1.5707963267948966) q[5];
y q[8];
tdg q[0];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
z q[0];
tdg q[3];
tdg q[1];
tdg q[10];
z q[5];
y q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[3];
u3(0, 0, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[4];
x q[9];
tdg q[0];
s q[9];
rz(1.5707963267948966) q[6];
h q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
x q[9];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[10];
sdg q[9];
h q[8];
s q[9];
x q[6];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[9];
y q[3];
x q[4];
s q[9];
y q[1];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[9];
t q[1];
t q[5];
tdg q[1];
x q[1];
z q[3];
tdg q[2];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
z q[3];
ry(1.5707963267948966) q[8];
tdg q[2];
x q[10];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[1];
tdg q[4];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[8];
s q[0];
sdg q[3];
sdg q[0];
tdg q[1];
sdg q[0];
t q[8];
x q[5];
x q[3];
t q[5];
y q[0];
h q[9];
h q[9];
tdg q[2];
y q[9];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
s q[7];
tdg q[4];
t q[10];
rz(1.5707963267948966) q[11];
s q[3];
h q[8];
z q[8];
x q[6];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
z q[4];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
s q[9];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[11];
x q[4];
tdg q[5];
tdg q[0];
u1(1.5707963267948966) q[10];
s q[6];
s q[6];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
x q[3];
sdg q[5];
s q[10];
h q[0];
rx(1.5707963267948966) q[7];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
s q[3];
sdg q[0];
y q[1];
u1(1.5707963267948966) q[9];
h q[4];
u3(0, 0, 1.5707963267948966) q[2];
y q[4];
u1(1.5707963267948966) q[8];
t q[5];
h q[5];
ry(1.5707963267948966) q[2];
z q[6];
h q[11];
u2(1.5707963267948966, 1.5707963267948966) q[5];
z q[5];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[0];
x q[6];
y q[1];
h q[0];
h q[7];
s q[5];
rz(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
x q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[7];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[8];
sdg q[10];
ry(1.5707963267948966) q[7];
h q[4];
s q[3];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[0];
rz(1.5707963267948966) q[2];
z q[0];
y q[3];
u1(1.5707963267948966) q[3];
y q[1];
rz(1.5707963267948966) q[7];
h q[3];
y q[11];
tdg q[4];
t q[4];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[7];
u3(0, 0, 1.5707963267948966) q[6];
s q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[4];
h q[1];
rx(1.5707963267948966) q[6];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[7];
sdg q[11];
tdg q[5];
h q[2];
s q[1];
y q[0];
tdg q[8];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
x q[3];
sdg q[8];
y q[6];
ry(1.5707963267948966) q[8];
y q[11];
t q[11];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[10];
x q[11];
sdg q[10];
ry(1.5707963267948966) q[11];
t q[7];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
y q[0];
h q[7];
y q[11];
u1(1.5707963267948966) q[4];
sdg q[11];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[3];
tdg q[11];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[2];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
tdg q[7];
sdg q[6];
rx(1.5707963267948966) q[8];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[10];
x q[2];
h q[9];
y q[9];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
y q[1];
z q[1];
h q[8];
u3(0, 0, 1.5707963267948966) q[11];
s q[3];
t q[1];
t q[1];
x q[4];
x q[7];
s q[8];
sdg q[11];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[0];
t q[9];
h q[3];
s q[3];
t q[8];
rx(1.5707963267948966) q[2];
z q[8];
h q[3];
rx(1.5707963267948966) q[2];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[10];
s q[6];
h q[6];
rx(1.5707963267948966) q[3];
s q[2];
x q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
sdg q[4];
tdg q[8];
t q[6];
tdg q[6];
u1(1.5707963267948966) q[0];
s q[4];
y q[0];
t q[6];
s q[4];
u1(1.5707963267948966) q[9];
z q[3];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[9];
tdg q[3];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
z q[0];
ry(1.5707963267948966) q[4];
s q[2];
y q[7];
sdg q[2];
t q[1];
rz(1.5707963267948966) q[3];
tdg q[9];
y q[5];
rx(1.5707963267948966) q[7];
t q[11];
sdg q[5];
sdg q[2];
h q[5];
z q[3];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[8];
tdg q[4];
x q[4];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[6];
tdg q[5];
y q[5];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[8];
tdg q[9];
rx(1.5707963267948966) q[7];
x q[10];
s q[8];
u3(0, 0, 1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
h q[2];
s q[5];
sdg q[1];
u1(1.5707963267948966) q[2];
y q[2];
s q[7];
s q[1];
s q[2];
u3(0, 0, 1.5707963267948966) q[11];
z q[1];
z q[6];
u1(1.5707963267948966) q[3];
tdg q[1];
rz(1.5707963267948966) q[9];
u1(1.5707963267948966) q[10];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
tdg q[10];
x q[2];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[9];
y q[3];
y q[6];
u3(0, 0, 1.5707963267948966) q[6];
sdg q[7];
sdg q[10];
ry(1.5707963267948966) q[8];
h q[2];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
s q[7];
z q[11];
h q[7];
rx(1.5707963267948966) q[2];
s q[1];
z q[11];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[10];
tdg q[6];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[2];
tdg q[9];
z q[2];
u3(0, 0, 1.5707963267948966) q[10];
tdg q[5];
rx(1.5707963267948966) q[11];
s q[8];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[7];
z q[6];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[3];
y q[11];
rz(1.5707963267948966) q[10];
sdg q[0];
tdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
z q[3];
rz(1.5707963267948966) q[5];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[11];
z q[5];
u1(1.5707963267948966) q[7];
y q[8];
z q[4];
tdg q[1];
y q[8];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[5];
t q[10];
sdg q[7];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[9];
tdg q[7];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
s q[11];
z q[0];
z q[1];
z q[5];
t q[1];
rz(1.5707963267948966) q[7];
y q[8];
rz(1.5707963267948966) q[10];
tdg q[7];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[4];
x q[1];
u1(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[7];
z q[9];
tdg q[10];
y q[1];
y q[10];
rz(1.5707963267948966) q[3];
s q[8];
t q[5];
sdg q[6];
z q[8];
ry(1.5707963267948966) q[6];
tdg q[5];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
s q[9];
rx(1.5707963267948966) q[2];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
sdg q[11];
sdg q[6];
h q[2];
z q[2];
s q[9];
rx(1.5707963267948966) q[11];
tdg q[0];
rx(1.5707963267948966) q[9];
sdg q[2];
y q[11];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
x q[3];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
t q[3];
y q[8];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[2];
