OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
ry(1.5707963267948966) q[1];
z q[2];
z q[11];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[5];
z q[7];
rx(1.5707963267948966) q[8];
u1(1.5707963267948966) q[11];
y q[7];
z q[10];
z q[6];
ry(1.5707963267948966) q[7];
y q[7];
sdg q[5];
t q[3];
u3(0, 0, 1.5707963267948966) q[8];
s q[7];
s q[1];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[11];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[10];
tdg q[7];
ry(1.5707963267948966) q[5];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
y q[4];
h q[1];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[2];
sdg q[2];
sdg q[2];
tdg q[7];
x q[3];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
z q[6];
s q[8];
u1(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[6];
x q[2];
s q[11];
y q[8];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[11];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[10];
t q[7];
x q[12];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[3];
s q[9];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[10];
ry(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[6];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[9];
s q[8];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
x q[5];
t q[5];
rx(1.5707963267948966) q[12];
t q[12];
z q[5];
rz(1.5707963267948966) q[10];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[12];
h q[10];
u3(0, 0, 1.5707963267948966) q[11];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[8];
y q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
sdg q[4];
h q[11];
y q[10];
t q[9];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[9];
y q[12];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
x q[12];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
z q[8];
u1(1.5707963267948966) q[10];
