OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
h q[8];
y q[12];
u2(1.5707963267948966, 1.5707963267948966) q[3];
y q[8];
h q[2];
u3(0, 0, 1.5707963267948966) q[6];
s q[11];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rx(1.5707963267948966) q[3];
x q[4];
x q[2];
y q[10];
t q[10];
h q[9];
z q[3];
u3(0, 0, 1.5707963267948966) q[4];
y q[1];
h q[3];
sdg q[0];
u1(1.5707963267948966) q[7];
t q[11];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[7];
z q[5];
x q[1];
s q[0];
z q[5];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
y q[3];
s q[12];
z q[7];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[10];
tdg q[6];
z q[12];
h q[8];
x q[12];
t q[11];
s q[1];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[9];
sdg q[11];
u3(0, 0, 1.5707963267948966) q[1];
z q[1];
h q[0];
y q[11];
z q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
z q[4];
rx(1.5707963267948966) q[11];
u1(1.5707963267948966) q[10];
z q[8];
h q[8];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
h q[12];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[0];
