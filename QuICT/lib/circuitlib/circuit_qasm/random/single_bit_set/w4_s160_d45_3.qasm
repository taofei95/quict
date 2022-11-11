OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
x q[2];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[2];
tdg q[3];
x q[0];
z q[2];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[0];
x q[2];
x q[2];
z q[3];
z q[3];
z q[2];
x q[0];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
sdg q[1];
rx(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[1];
x q[1];
h q[2];
t q[2];
rz(1.5707963267948966) q[0];
t q[3];
sdg q[3];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[1];
y q[1];
tdg q[2];
t q[0];
s q[2];
sdg q[3];
t q[2];
tdg q[0];
s q[1];
u1(1.5707963267948966) q[3];
x q[3];
s q[3];
u3(0, 0, 1.5707963267948966) q[1];
h q[1];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[0];
x q[1];
rx(1.5707963267948966) q[3];
tdg q[3];
s q[1];
rz(1.5707963267948966) q[2];
t q[3];
s q[3];
s q[1];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[3];
rx(1.5707963267948966) q[3];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[0];
rx(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
y q[1];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[1];
u1(1.5707963267948966) q[3];
t q[0];
z q[2];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[2];
x q[1];
s q[0];
u3(0, 0, 1.5707963267948966) q[0];
h q[1];
u3(0, 0, 1.5707963267948966) q[2];
x q[2];
y q[2];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
x q[1];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[0];
s q[1];
u1(1.5707963267948966) q[0];
s q[0];
z q[0];
rz(1.5707963267948966) q[2];
z q[0];
sdg q[3];
x q[2];
h q[0];
x q[0];
z q[2];
y q[0];
tdg q[0];
tdg q[0];
x q[2];
y q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[1];
ry(1.5707963267948966) q[1];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[0];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[3];
tdg q[0];
h q[3];
sdg q[0];
rz(1.5707963267948966) q[1];
z q[0];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
x q[2];
u1(1.5707963267948966) q[1];
y q[3];
x q[2];
h q[3];
s q[2];
s q[0];
u3(0, 0, 1.5707963267948966) q[0];
s q[2];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[3];
z q[1];
z q[3];
y q[1];
tdg q[3];
s q[1];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
x q[3];
t q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[1];
h q[3];
