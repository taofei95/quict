OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
h q[10];
cx q[0], q[5];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
h q[7];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
cx q[1], q[9];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
cx q[2], q[6];
cx q[5], q[4];
cx q[10], q[2];
cx q[9], q[4];
cx q[1], q[9];
cx q[6], q[11];
cx q[7], q[10];
rx(1.5707963267948966) q[11];
h q[9];
rx(1.5707963267948966) q[1];
h q[8];
ry(1.5707963267948966) q[0];
cx q[0], q[7];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
h q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
h q[1];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
cx q[0], q[8];
ry(1.5707963267948966) q[9];
cx q[11], q[4];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[1];
h q[8];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
cx q[3], q[8];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[11];
