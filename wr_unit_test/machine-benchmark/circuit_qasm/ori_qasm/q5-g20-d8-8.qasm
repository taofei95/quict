OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
z q[3];
rx(1.5707963267948966) q[3];
t q[1];
h q[4];
sx q[1];
swap q[2], q[1];
t q[2];
ry(1.5707963267948966) q[2];
x q[4];
swap q[2], q[1];
t q[4];
y q[2];
h q[1];
sx q[0];
y q[3];
h q[3];
z q[4];
t q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];