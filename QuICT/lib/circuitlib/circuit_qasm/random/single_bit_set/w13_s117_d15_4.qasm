OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
x q[8];
rx(1.5707963267948966) q[7];
u1(1.5707963267948966) q[1];
t q[11];
h q[1];
s q[2];
s q[4];
u1(1.5707963267948966) q[11];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[2];
s q[1];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[6];
u1(1.5707963267948966) q[3];
t q[10];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[0];
t q[1];
s q[10];
x q[7];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[0];
h q[5];
t q[4];
tdg q[6];
sdg q[5];
x q[5];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
h q[7];
y q[3];
h q[2];
u1(1.5707963267948966) q[0];
t q[12];
tdg q[12];
y q[11];
y q[8];
y q[12];
tdg q[12];
rz(1.5707963267948966) q[8];
x q[10];
x q[2];
z q[11];
rz(1.5707963267948966) q[4];
h q[4];
rx(1.5707963267948966) q[10];
s q[2];
x q[8];
rz(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[8];
h q[6];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[6];
tdg q[6];
sdg q[0];
z q[2];
sdg q[11];
x q[8];
y q[11];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
x q[8];
ry(1.5707963267948966) q[1];
x q[4];
x q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[12];
z q[4];
rx(1.5707963267948966) q[11];
s q[0];
t q[7];
u1(1.5707963267948966) q[10];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[7];
t q[3];
z q[5];
y q[11];
tdg q[7];
sdg q[3];
rx(1.5707963267948966) q[9];
h q[10];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[7];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[12];
z q[1];
u1(1.5707963267948966) q[12];
rx(1.5707963267948966) q[10];
x q[11];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[3];
tdg q[2];
h q[4];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[4];
tdg q[12];
u1(1.5707963267948966) q[12];
ry(1.5707963267948966) q[2];
y q[0];
tdg q[4];
h q[9];
z q[3];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
s q[1];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[2];
