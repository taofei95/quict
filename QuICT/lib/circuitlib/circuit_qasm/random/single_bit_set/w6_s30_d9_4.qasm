OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
t q[2];
u3(0, 0, 1.5707963267948966) q[3];
x q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
z q[1];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
t q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[4];
h q[4];
h q[2];
ry(1.5707963267948966) q[1];
sdg q[4];
rz(1.5707963267948966) q[1];
z q[3];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
s q[4];
h q[0];
h q[5];
u1(1.5707963267948966) q[3];
z q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
y q[4];
ry(1.5707963267948966) q[4];
z q[1];
y q[5];
