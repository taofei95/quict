OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[4];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
sdg q[0];
s q[4];
t q[0];
sdg q[3];
ry(1.5707963267948966) q[2];
s q[4];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
tdg q[0];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
h q[1];
x q[1];
rx(1.5707963267948966) q[2];
t q[0];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[1];
y q[0];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[1];
ry(1.5707963267948966) q[1];
