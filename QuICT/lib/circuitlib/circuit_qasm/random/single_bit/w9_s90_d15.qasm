OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
sdg q[7];
h q[4];
s q[1];
s q[7];
t q[6];
sdg q[4];
x q[5];
x q[4];
h q[6];
tdg q[1];
sdg q[5];
rz(1.5707963267948966) q[2];
tdg q[3];
z q[5];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
s q[2];
y q[6];
y q[3];
z q[8];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[4];
rx(1.5707963267948966) q[0];
sdg q[6];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[2];
tdg q[8];
h q[2];
t q[6];
tdg q[7];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
tdg q[2];
s q[4];
x q[6];
s q[8];
s q[5];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[2];
x q[3];
tdg q[1];
t q[7];
u1(1.5707963267948966) q[2];
x q[8];
s q[5];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
z q[2];
s q[6];
u3(0, 0, 1.5707963267948966) q[8];
h q[1];
x q[6];
u3(0, 0, 1.5707963267948966) q[0];
y q[5];
s q[2];
rx(1.5707963267948966) q[2];
z q[3];
z q[8];
u3(0, 0, 1.5707963267948966) q[4];
t q[5];
t q[1];
sdg q[5];
s q[1];
x q[5];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[8];
s q[4];
t q[0];
tdg q[6];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[4];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
y q[3];
t q[5];
t q[3];
u3(0, 0, 1.5707963267948966) q[1];
s q[7];
sdg q[2];
h q[8];
tdg q[1];
y q[6];
u3(0, 0, 1.5707963267948966) q[1];
