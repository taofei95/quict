OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rx(1.5707963267948966) q[1];
cx q[1], q[0];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
cx q[0], q[1];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
h q[1];
h q[0];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
h q[1];
rx(1.5707963267948966) q[1];
h q[2];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
cx q[1], q[2];
h q[0];
rx(1.5707963267948966) q[0];
h q[2];
h q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
h q[0];
h q[0];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
cx q[1], q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
cx q[2], q[0];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
h q[2];
h q[0];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
h q[0];
cx q[1], q[2];
ry(1.5707963267948966) q[1];
cx q[0], q[2];
ry(1.5707963267948966) q[1];
cx q[2], q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
cx q[2], q[1];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
h q[0];
cx q[0], q[2];
h q[2];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
cx q[2], q[1];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
cx q[2], q[0];
h q[2];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
h q[0];
cx q[0], q[1];
h q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
h q[2];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
h q[2];
ry(1.5707963267948966) q[0];
