OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[2];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[2];
y q[3];
z q[4];
rx(1.5707963267948966) q[2];
z q[3];
x q[4];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[2];
rx(1.5707963267948966) q[1];
s q[3];
h q[4];
x q[3];
sdg q[2];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
x q[0];
y q[1];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[1];
h q[3];
h q[3];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[1];
t q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
t q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
x q[1];
t q[3];
s q[0];
