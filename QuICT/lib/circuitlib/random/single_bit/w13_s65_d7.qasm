OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
h q[2];
s q[12];
h q[6];
x q[4];
u3(0, 0, 1.5707963267948966) q[11];
t q[5];
t q[10];
t q[0];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[5];
x q[10];
s q[1];
sdg q[0];
z q[0];
ry(1.5707963267948966) q[3];
h q[4];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[6];
z q[8];
u1(1.5707963267948966) q[7];
x q[1];
z q[0];
y q[6];
ry(1.5707963267948966) q[7];
t q[11];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[4];
s q[8];
ry(1.5707963267948966) q[0];
tdg q[10];
t q[7];
u3(0, 0, 1.5707963267948966) q[7];
x q[6];
x q[12];
rz(1.5707963267948966) q[9];
s q[11];
s q[12];
ry(1.5707963267948966) q[8];
s q[8];
rz(1.5707963267948966) q[11];
h q[8];
u2(1.5707963267948966, 1.5707963267948966) q[4];
x q[9];
sdg q[3];
x q[1];
h q[9];
tdg q[3];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[12];
ry(1.5707963267948966) q[11];
y q[11];
h q[10];
sdg q[0];
t q[1];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[4];