OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
y q[9];
s q[1];
tdg q[9];
sdg q[6];
u1(1.5707963267948966) q[4];
z q[3];
sdg q[4];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[0];
z q[4];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
h q[10];
t q[2];
x q[5];
x q[8];
s q[10];
z q[6];
z q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[9];
sdg q[9];
tdg q[0];
t q[11];
sdg q[0];
rz(1.5707963267948966) q[5];
sdg q[2];
y q[4];
s q[1];
x q[9];
z q[11];
u1(1.5707963267948966) q[10];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
y q[2];
t q[8];
sdg q[5];
h q[7];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[10];
sdg q[11];
ry(1.5707963267948966) q[8];
tdg q[10];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
y q[5];
y q[0];
rx(1.5707963267948966) q[0];
t q[7];
s q[4];
x q[1];
z q[7];
s q[10];
h q[0];
