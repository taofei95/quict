OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
h q[0];
cx q[0], q[1];
h q[3];
cx q[0], q[3];
h q[5];
cx q[2], q[1];
ry(1.5707963267948966) q[2];
h q[5];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
h q[3];
h q[1];
h q[2];
cx q[4], q[5];
h q[2];
cx q[2], q[3];
ry(1.5707963267948966) q[2];
cx q[3], q[5];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
h q[3];
h q[4];
rz(1.5707963267948966) q[5];
h q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
cx q[3], q[2];
cx q[5], q[0];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[0];
