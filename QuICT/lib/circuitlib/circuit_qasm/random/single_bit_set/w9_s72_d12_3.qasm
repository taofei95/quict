OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
rx(1.5707963267948966) q[3];
y q[6];
s q[4];
sdg q[4];
h q[6];
sdg q[4];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[3];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
t q[2];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
tdg q[7];
rz(1.5707963267948966) q[5];
y q[0];
rz(1.5707963267948966) q[5];
sdg q[0];
h q[1];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
y q[1];
u1(1.5707963267948966) q[5];
y q[0];
rx(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[1];
z q[8];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[3];
u1(1.5707963267948966) q[1];
t q[0];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[2];
tdg q[3];
z q[0];
tdg q[2];
rz(1.5707963267948966) q[6];
y q[1];
rx(1.5707963267948966) q[6];
s q[8];
sdg q[3];
ry(1.5707963267948966) q[5];
s q[0];
tdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
tdg q[4];
y q[0];
sdg q[5];
tdg q[2];
ry(1.5707963267948966) q[8];
y q[1];
t q[6];
u3(0, 0, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
x q[8];
z q[1];
sdg q[3];
tdg q[1];
