OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rx(1.5707963267948966) q[5];
tdg q[2];
ry(1.5707963267948966) q[4];
sdg q[5];
x q[5];
s q[0];
y q[5];
u3(0, 0, 1.5707963267948966) q[3];
z q[2];
u3(0, 0, 1.5707963267948966) q[5];
h q[2];
x q[2];
y q[2];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[2];
s q[0];
h q[5];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
x q[0];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
s q[4];
tdg q[0];
sdg q[5];
u1(1.5707963267948966) q[1];
tdg q[1];
sdg q[1];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
z q[3];
t q[4];
rx(1.5707963267948966) q[1];
y q[2];
h q[4];
t q[4];
u3(0, 0, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[2];
h q[5];
s q[0];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[3];
tdg q[1];
tdg q[1];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
h q[4];
x q[0];
tdg q[3];