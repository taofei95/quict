OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cx q[3], q[0];
cx q[2], q[7];
h q[5];
cx q[2], q[3];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
cx q[4], q[3];
rz(1.5707963267948966) q[3];
cx q[1], q[4];
cx q[8], q[7];
h q[0];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[8];
cx q[1], q[0];
rz(1.5707963267948966) q[8];
h q[4];
rz(1.5707963267948966) q[3];
h q[0];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[7];
cx q[7], q[2];
cx q[1], q[5];
cx q[5], q[4];
cx q[6], q[2];
rz(1.5707963267948966) q[5];
cx q[8], q[6];
h q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
cx q[4], q[5];
h q[6];
ry(1.5707963267948966) q[4];
h q[4];
h q[8];
cx q[5], q[2];
cx q[4], q[3];
rx(1.5707963267948966) q[0];
cx q[4], q[3];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[1];
h q[3];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
cx q[0], q[4];
rx(1.5707963267948966) q[6];
h q[2];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
h q[3];
rz(1.5707963267948966) q[2];
cx q[0], q[6];
cx q[0], q[3];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
cx q[7], q[5];
h q[2];
cx q[1], q[8];
rz(1.5707963267948966) q[3];
cx q[1], q[2];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
h q[7];
ry(1.5707963267948966) q[0];
cx q[7], q[4];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[7];
cx q[6], q[0];
ry(1.5707963267948966) q[3];
h q[1];
h q[4];
h q[4];
cx q[5], q[8];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
cx q[1], q[6];
h q[3];
h q[7];
cx q[5], q[8];
h q[0];
rx(1.5707963267948966) q[2];
cx q[7], q[4];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
h q[5];
cx q[6], q[0];
cx q[8], q[5];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[8];
cx q[6], q[1];
cx q[1], q[8];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
h q[3];
rx(1.5707963267948966) q[4];
cx q[8], q[2];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[7];
h q[7];
rx(1.5707963267948966) q[2];
h q[1];
rz(1.5707963267948966) q[5];
h q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
h q[5];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[8];
cx q[6], q[2];
rx(1.5707963267948966) q[2];
h q[8];
rx(1.5707963267948966) q[7];
cx q[7], q[8];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
h q[6];
ry(1.5707963267948966) q[0];
h q[4];
h q[2];
rz(1.5707963267948966) q[4];
cx q[2], q[5];
ry(1.5707963267948966) q[8];
cx q[1], q[8];
h q[6];
h q[6];
h q[0];
h q[3];
cx q[8], q[5];
h q[4];
h q[5];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
h q[4];
h q[1];
h q[5];
h q[3];
cx q[0], q[3];
rz(1.5707963267948966) q[6];
h q[2];
cx q[4], q[1];
cx q[8], q[1];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[5];
h q[0];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
cx q[8], q[7];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
cx q[6], q[5];
cx q[1], q[7];
cx q[0], q[4];
rz(1.5707963267948966) q[6];
cx q[8], q[6];
cx q[7], q[6];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[8];
cx q[5], q[2];
h q[6];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[3];
