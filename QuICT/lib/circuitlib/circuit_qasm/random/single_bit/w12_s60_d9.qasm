OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
z q[9];
tdg q[3];
sdg q[9];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[8];
z q[0];
x q[7];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[11];
u1(1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
tdg q[6];
y q[10];
s q[0];
ry(1.5707963267948966) q[6];
x q[1];
s q[8];
s q[7];
x q[10];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
y q[2];
u3(0, 0, 1.5707963267948966) q[9];
t q[6];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
h q[2];
s q[2];
s q[10];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[11];
x q[7];
ry(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[11];
x q[1];
sdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[11];
t q[10];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[8];
h q[10];
z q[11];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[5];
h q[8];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[7];
x q[2];
t q[8];
rx(1.5707963267948966) q[11];
sdg q[5];
rz(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[8];
