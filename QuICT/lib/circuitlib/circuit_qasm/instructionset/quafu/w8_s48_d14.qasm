OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
cx q[2], q[1];
h q[5];
rz(1.5707963267948966) q[0];
h q[4];
rx(1.5707963267948966) q[6];
h q[6];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
h q[5];
h q[6];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
cx q[1], q[7];
cx q[4], q[6];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
cx q[0], q[1];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
h q[5];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[7];
h q[5];
cx q[1], q[7];
h q[4];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[7];
cx q[3], q[1];
rz(1.5707963267948966) q[4];
h q[0];
cx q[1], q[4];
cx q[1], q[4];
cx q[0], q[3];
h q[1];
ry(1.5707963267948966) q[5];
cx q[4], q[5];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
cx q[5], q[4];
rz(1.5707963267948966) q[0];
