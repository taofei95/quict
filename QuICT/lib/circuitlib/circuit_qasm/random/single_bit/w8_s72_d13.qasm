OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rx(1.5707963267948966) q[5];
h q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
sdg q[5];
t q[0];
t q[1];
x q[3];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[2];
s q[7];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[6];
h q[5];
z q[1];
y q[2];
z q[5];
u3(0, 0, 1.5707963267948966) q[0];
x q[0];
u1(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
x q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
x q[2];
h q[5];
s q[0];
z q[6];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[3];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[2];
t q[6];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[4];
h q[5];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[5];
x q[5];
rx(1.5707963267948966) q[3];
tdg q[3];
rx(1.5707963267948966) q[3];
x q[4];
u1(1.5707963267948966) q[5];
t q[1];
t q[0];
z q[0];
sdg q[1];
rz(1.5707963267948966) q[1];
h q[4];
tdg q[0];
x q[5];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[4];
rz(1.5707963267948966) q[6];
u1(1.5707963267948966) q[7];
y q[6];
x q[5];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[0];
sdg q[6];
z q[6];
