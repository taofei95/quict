OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
rz(1.5707963267948966) q[12];
sdg q[9];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
h q[12];
rx(1.5707963267948966) q[1];
u1(1.5707963267948966) q[6];
t q[8];
tdg q[4];
t q[0];
u1(1.5707963267948966) q[12];
z q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[0];
ry(1.5707963267948966) q[1];
sdg q[3];
sdg q[6];
h q[4];
h q[9];
x q[1];
t q[2];
rx(1.5707963267948966) q[11];
y q[11];
u3(0, 0, 1.5707963267948966) q[7];
t q[1];
z q[9];
sdg q[1];
x q[3];
rz(1.5707963267948966) q[8];
z q[7];
rz(1.5707963267948966) q[1];
x q[0];
x q[1];
rz(1.5707963267948966) q[5];
s q[3];
rx(1.5707963267948966) q[10];
x q[5];
rx(1.5707963267948966) q[11];
x q[11];
z q[6];
u3(0, 0, 1.5707963267948966) q[4];
t q[3];
rx(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[5];
z q[9];
h q[5];
t q[12];
s q[2];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[11];
s q[9];
t q[7];
t q[4];
t q[2];
rx(1.5707963267948966) q[10];
tdg q[3];
rx(1.5707963267948966) q[11];
sdg q[3];
sdg q[0];
tdg q[1];
z q[0];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
x q[12];
sdg q[10];
x q[7];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
z q[10];
h q[8];
u1(1.5707963267948966) q[0];
s q[4];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
tdg q[0];
tdg q[6];
s q[10];
u3(0, 0, 1.5707963267948966) q[7];
sdg q[1];
x q[10];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[9];
t q[3];
t q[11];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[4];
sdg q[12];
u1(1.5707963267948966) q[11];
h q[12];
u3(0, 0, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
x q[7];
h q[10];
h q[10];
h q[9];
u2(1.5707963267948966, 1.5707963267948966) q[6];
z q[6];
rz(1.5707963267948966) q[12];
y q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[10];
tdg q[12];
y q[4];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[10];
y q[10];
h q[4];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[1];
rz(1.5707963267948966) q[0];
x q[2];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[3];
x q[0];
rz(1.5707963267948966) q[10];
y q[9];
z q[9];
y q[11];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[12];
y q[1];
t q[0];
s q[12];
ry(1.5707963267948966) q[8];
sdg q[10];
u1(1.5707963267948966) q[8];
rx(1.5707963267948966) q[4];
tdg q[1];
t q[10];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
s q[6];
tdg q[0];
