OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rz(1.5707963267948966) q[5];
z q[2];
t q[4];
h q[2];
s q[7];
sdg q[6];
ry(1.5707963267948966) q[1];
t q[1];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
h q[0];
s q[4];
tdg q[0];
sdg q[0];
y q[3];
u3(0, 0, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[4];
z q[1];
tdg q[0];
sdg q[0];
ry(1.5707963267948966) q[1];
t q[2];
ry(1.5707963267948966) q[7];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[6];
y q[4];
s q[4];
y q[7];
u3(0, 0, 1.5707963267948966) q[6];
tdg q[7];
rz(1.5707963267948966) q[0];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[6];
t q[2];
x q[5];
ry(1.5707963267948966) q[5];
h q[1];
s q[6];
h q[7];
x q[1];
sdg q[1];
tdg q[0];
ry(1.5707963267948966) q[4];
z q[0];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
s q[7];
s q[3];
z q[6];
y q[2];
x q[0];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
x q[7];
u3(0, 0, 1.5707963267948966) q[5];
z q[5];
rz(1.5707963267948966) q[7];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[1];
t q[2];
z q[3];
s q[5];
t q[2];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[3];
sdg q[0];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
sdg q[6];
y q[0];
t q[4];
sdg q[5];
y q[2];
h q[5];
tdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[1];
z q[5];
tdg q[7];
u1(1.5707963267948966) q[4];
s q[4];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
z q[7];
