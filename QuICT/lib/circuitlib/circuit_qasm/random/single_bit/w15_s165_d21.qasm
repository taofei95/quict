OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[8];
s q[9];
rx(1.5707963267948966) q[5];
tdg q[7];
t q[9];
u3(0, 0, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[13];
x q[2];
s q[8];
rz(1.5707963267948966) q[9];
y q[6];
tdg q[3];
y q[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[5];
y q[3];
ry(1.5707963267948966) q[0];
x q[7];
rx(1.5707963267948966) q[13];
x q[13];
z q[8];
sdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[10];
x q[11];
s q[1];
u3(0, 0, 1.5707963267948966) q[2];
t q[7];
tdg q[13];
y q[2];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[6];
h q[7];
rx(1.5707963267948966) q[12];
z q[1];
x q[6];
u1(1.5707963267948966) q[7];
t q[6];
z q[12];
tdg q[0];
rz(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[12];
rz(1.5707963267948966) q[12];
z q[3];
u3(0, 0, 1.5707963267948966) q[8];
y q[4];
h q[0];
u1(1.5707963267948966) q[9];
tdg q[0];
ry(1.5707963267948966) q[2];
t q[9];
ry(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
x q[6];
y q[12];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[7];
x q[2];
u3(0, 0, 1.5707963267948966) q[11];
rz(1.5707963267948966) q[12];
s q[1];
y q[6];
tdg q[8];
h q[0];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
x q[4];
u1(1.5707963267948966) q[2];
t q[0];
sdg q[13];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[8];
t q[7];
x q[1];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[10];
sdg q[6];
t q[4];
s q[6];
t q[2];
tdg q[8];
x q[4];
ry(1.5707963267948966) q[1];
t q[6];
z q[8];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
z q[11];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[4];
rx(1.5707963267948966) q[0];
h q[11];
sdg q[12];
y q[4];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
s q[7];
h q[2];
u3(0, 0, 1.5707963267948966) q[12];
y q[13];
rz(1.5707963267948966) q[9];
z q[7];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[3];
sdg q[14];
x q[4];
sdg q[14];
z q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
y q[6];
ry(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[5];
rx(1.5707963267948966) q[7];
t q[8];
u3(0, 0, 1.5707963267948966) q[4];
z q[5];
tdg q[6];
y q[2];
x q[8];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[9];
z q[0];
h q[3];
s q[6];
z q[8];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
x q[5];
rx(1.5707963267948966) q[7];
x q[6];
u3(0, 0, 1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[3];
sdg q[7];
y q[13];
y q[10];
y q[6];
u3(0, 0, 1.5707963267948966) q[10];
s q[11];
rx(1.5707963267948966) q[1];
h q[14];
rz(1.5707963267948966) q[9];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[7];
h q[9];