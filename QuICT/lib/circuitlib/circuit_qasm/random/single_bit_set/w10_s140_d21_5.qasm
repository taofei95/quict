OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
s q[3];
h q[6];
z q[8];
t q[0];
y q[0];
z q[7];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[8];
t q[6];
h q[9];
s q[7];
rz(1.5707963267948966) q[5];
h q[5];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[7];
u1(1.5707963267948966) q[7];
z q[8];
x q[9];
ry(1.5707963267948966) q[7];
tdg q[8];
sdg q[9];
t q[1];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[7];
t q[4];
s q[7];
u3(0, 0, 1.5707963267948966) q[1];
x q[3];
u1(1.5707963267948966) q[4];
h q[2];
sdg q[2];
ry(1.5707963267948966) q[1];
t q[6];
sdg q[7];
sdg q[2];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[5];
tdg q[6];
s q[6];
x q[2];
rx(1.5707963267948966) q[5];
t q[7];
s q[1];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[8];
x q[1];
y q[4];
h q[2];
x q[0];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[9];
tdg q[2];
sdg q[9];
x q[3];
y q[0];
y q[1];
t q[1];
sdg q[3];
z q[1];
rx(1.5707963267948966) q[9];
h q[8];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[1];
sdg q[5];
tdg q[2];
s q[6];
h q[1];
y q[7];
u3(0, 0, 1.5707963267948966) q[2];
y q[4];
sdg q[5];
rx(1.5707963267948966) q[7];
tdg q[6];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[2];
sdg q[3];
u3(0, 0, 1.5707963267948966) q[7];
sdg q[7];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[3];
h q[6];
sdg q[3];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[7];
sdg q[0];
rx(1.5707963267948966) q[2];
x q[0];
rx(1.5707963267948966) q[4];
x q[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
y q[7];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
t q[4];
u1(1.5707963267948966) q[8];
sdg q[1];
y q[7];
s q[6];
sdg q[5];
s q[6];
z q[2];
x q[8];
tdg q[0];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[1];
z q[1];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
sdg q[4];
x q[3];
rx(1.5707963267948966) q[9];
sdg q[0];
h q[7];
h q[7];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[0];
u1(1.5707963267948966) q[2];
tdg q[8];
u3(0, 0, 1.5707963267948966) q[4];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[6];
z q[5];
z q[8];
t q[0];
x q[9];
z q[3];
t q[5];
rx(1.5707963267948966) q[1];
