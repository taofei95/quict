OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[0];
tdg q[3];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[8];
z q[0];
h q[0];
ry(1.5707963267948966) q[8];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[3];
y q[3];
z q[1];
x q[8];
y q[3];
h q[4];
s q[3];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
x q[6];
z q[3];
x q[2];
z q[8];
h q[4];
z q[7];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
t q[7];
u3(0, 0, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[5];
z q[2];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[1];
y q[6];
y q[0];
sdg q[3];
s q[0];
z q[2];
s q[7];
x q[6];
y q[5];
sdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
y q[2];
x q[6];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
t q[6];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[1];
s q[3];
tdg q[1];
rx(1.5707963267948966) q[4];
h q[0];