OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
sdg q[6];
tdg q[3];
s q[2];
tdg q[2];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[1];
s q[0];
h q[7];
sdg q[6];
h q[3];
t q[0];
y q[6];
z q[1];
rz(1.5707963267948966) q[3];
z q[1];
h q[4];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[8];
z q[5];
h q[0];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[6];
s q[5];
tdg q[2];
t q[0];
t q[5];
tdg q[8];
z q[5];
y q[6];
h q[3];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[6];
h q[8];
x q[0];
rz(1.5707963267948966) q[4];
x q[7];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[8];
h q[4];
sdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[5];
ry(1.5707963267948966) q[5];
y q[1];
tdg q[7];
u1(1.5707963267948966) q[7];
x q[2];
s q[7];
y q[8];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[3];
h q[7];
u3(0, 0, 1.5707963267948966) q[5];
x q[5];
x q[0];
z q[1];
rz(1.5707963267948966) q[0];
h q[2];
t q[7];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
x q[5];
sdg q[6];
rx(1.5707963267948966) q[2];
s q[0];
ry(1.5707963267948966) q[3];
z q[1];
s q[7];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
t q[4];
t q[1];
sdg q[6];
tdg q[8];
y q[7];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[0];
u1(1.5707963267948966) q[7];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
z q[6];
sdg q[5];
rx(1.5707963267948966) q[1];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[0];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
s q[8];
rz(1.5707963267948966) q[3];
z q[6];
u3(0, 0, 1.5707963267948966) q[4];
y q[0];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[4];
