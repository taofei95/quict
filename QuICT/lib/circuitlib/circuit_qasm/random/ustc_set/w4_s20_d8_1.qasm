OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
cx q[1], q[0];
h q[3];
ry(1.5707963267948966) q[3];
h q[3];
x q[1];
x q[0];
rx(1.5707963267948966) q[3];
h q[3];
rz(1.5707963267948966) q[0];
cx q[2], q[0];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
h q[0];
h q[3];
h q[1];
cx q[2], q[3];
