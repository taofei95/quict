OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
cx q[5], q[0];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
h q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
cx q[0], q[5];
cx q[5], q[2];
h q[6];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
cx q[3], q[5];
cx q[2], q[5];
cx q[4], q[1];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
cx q[5], q[4];
cx q[0], q[2];
rz(1.5707963267948966) q[6];
h q[3];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
cx q[6], q[2];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
cx q[0], q[6];
rx(1.5707963267948966) q[2];
cx q[1], q[0];
cx q[5], q[3];
cx q[2], q[6];
rx(1.5707963267948966) q[4];
h q[4];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
h q[2];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
h q[4];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
cx q[2], q[4];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
cx q[1], q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
h q[6];
h q[1];
h q[1];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
cx q[6], q[0];
h q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[4];
cx q[4], q[2];
cx q[1], q[4];
h q[2];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
h q[1];
h q[3];
rz(1.5707963267948966) q[1];
cx q[5], q[2];
h q[1];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
h q[3];
ry(1.5707963267948966) q[4];
h q[5];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
h q[1];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
cx q[6], q[0];
h q[2];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
cx q[4], q[2];
cx q[3], q[2];
h q[1];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
h q[5];
cx q[6], q[4];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
h q[0];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
h q[0];
rx(1.5707963267948966) q[0];
h q[3];
h q[4];
cx q[5], q[6];
cx q[6], q[4];
cx q[3], q[6];
h q[2];
rx(1.5707963267948966) q[2];
h q[2];
cx q[3], q[2];
cx q[0], q[1];
h q[0];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
h q[3];
