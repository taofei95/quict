OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
sdg q[5];
x q[0];
h q[0];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[12];
rx(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[11];
rx(1.5707963267948966) q[5];
s q[10];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[10];
h q[3];
ry(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[12];
x q[5];
ry(1.5707963267948966) q[0];
s q[9];
y q[1];
h q[2];
x q[4];
rz(1.5707963267948966) q[6];
y q[6];
h q[8];
u3(0, 0, 1.5707963267948966) q[9];
s q[8];
rx(1.5707963267948966) q[11];
u1(1.5707963267948966) q[11];
u1(1.5707963267948966) q[11];
z q[7];
rx(1.5707963267948966) q[6];
h q[0];
t q[6];
ry(1.5707963267948966) q[4];
x q[11];
s q[6];
rx(1.5707963267948966) q[8];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[10];
sdg q[7];
sdg q[3];
rx(1.5707963267948966) q[6];
y q[4];
sdg q[11];
x q[8];
rz(1.5707963267948966) q[1];
t q[12];
tdg q[11];
u1(1.5707963267948966) q[5];
s q[8];
y q[8];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[10];
sdg q[9];
h q[11];
ry(1.5707963267948966) q[6];
s q[4];
t q[8];
sdg q[1];
x q[2];
u3(0, 0, 1.5707963267948966) q[6];
h q[5];
h q[10];
s q[2];
x q[3];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[10];