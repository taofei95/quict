OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
y q[3];
h q[0];
s q[3];
t q[0];
h q[3];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
h q[6];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[1];
x q[2];
rz(1.5707963267948966) q[2];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
x q[3];
z q[3];
t q[0];
sdg q[0];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[3];
sdg q[1];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[3];
s q[0];
sdg q[1];
rx(1.5707963267948966) q[2];
tdg q[0];
ry(1.5707963267948966) q[3];
tdg q[6];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[3];
y q[2];
tdg q[0];
x q[2];
tdg q[3];
x q[2];
h q[1];
rx(1.5707963267948966) q[5];
s q[6];
u1(1.5707963267948966) q[6];
sdg q[3];
ry(1.5707963267948966) q[6];
sdg q[0];
x q[1];
ry(1.5707963267948966) q[3];
y q[3];
x q[2];
rz(1.5707963267948966) q[4];
z q[1];
h q[0];
h q[1];
h q[6];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
t q[6];
t q[4];
u1(1.5707963267948966) q[3];
h q[3];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
s q[2];
t q[2];
tdg q[1];
y q[4];
h q[1];
s q[6];
t q[4];
x q[2];
ry(1.5707963267948966) q[5];
tdg q[4];
