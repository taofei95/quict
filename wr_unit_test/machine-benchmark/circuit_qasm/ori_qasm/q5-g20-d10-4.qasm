OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[0];
h q[2];
y q[4];
cx q[2], q[0];
sdg q[0];
rx(1.5707963267948966) q[1];
h q[4];
rx(1.5707963267948966) q[1];
x q[1];
rz(1.5707963267948966) q[0];
z q[0];
t q[1];
ry(1.5707963267948966) q[0];
s q[4];
t q[4];
y q[0];
x q[4];
sdg q[0];
swap q[2], q[0];
h q[0];