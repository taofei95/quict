OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];
sdg q[7];
tdg q[3];
z q[4];
s q[8];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[7];
s q[0];
rx(1.5707963267948966) q[8];
u1(1.5707963267948966) q[5];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[8];
t q[12];
u1(1.5707963267948966) q[13];
z q[9];
t q[9];
s q[6];
t q[10];
y q[7];
y q[3];
u1(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[11];
ry(1.5707963267948966) q[12];
sdg q[13];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[10];
x q[9];
h q[13];
y q[13];
ry(1.5707963267948966) q[13];
y q[6];
h q[9];
z q[1];
u3(0, 0, 1.5707963267948966) q[0];
h q[3];
t q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[0];
z q[9];
u3(0, 0, 1.5707963267948966) q[7];
tdg q[7];
sdg q[4];
t q[12];
s q[7];
x q[0];
ry(1.5707963267948966) q[10];
z q[6];
y q[4];
s q[3];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[5];
y q[3];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[8];
y q[2];
sdg q[5];
rz(1.5707963267948966) q[6];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[10];
y q[7];
s q[8];
tdg q[8];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[8];
ry(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[12];
z q[13];
ry(1.5707963267948966) q[11];
sdg q[5];
tdg q[7];
u1(1.5707963267948966) q[7];
x q[8];
z q[5];
s q[3];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[12];
h q[8];
rz(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[6];
z q[3];
u1(1.5707963267948966) q[8];
z q[13];
u1(1.5707963267948966) q[3];
h q[2];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[12];
ry(1.5707963267948966) q[0];
t q[7];
u3(0, 0, 1.5707963267948966) q[8];
x q[6];
tdg q[1];
s q[1];
ry(1.5707963267948966) q[2];
t q[9];
t q[2];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[13];
s q[11];
sdg q[13];
sdg q[13];
u3(0, 0, 1.5707963267948966) q[5];
