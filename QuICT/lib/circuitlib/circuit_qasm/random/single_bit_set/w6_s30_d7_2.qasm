OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
z q[2];
sdg q[3];
tdg q[2];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
tdg q[5];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
sdg q[2];
s q[1];
y q[0];
s q[1];
h q[3];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
s q[4];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
x q[1];
y q[1];
u3(0, 0, 1.5707963267948966) q[4];
z q[3];
z q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
