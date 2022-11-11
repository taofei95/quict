OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[10];
x q[8];
rx(1.5707963267948966) q[4];
cx q[4], q[5];
cx q[0], q[1];
x q[2];
cx q[4], q[6];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[3];
cx q[6], q[5];
ry(1.5707963267948966) q[5];
cx q[4], q[6];
cx q[7], q[3];
h q[9];
x q[10];
ry(1.5707963267948966) q[2];
x q[2];
rz(1.5707963267948966) q[1];
x q[6];
h q[6];
cx q[3], q[5];
h q[3];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[10];
x q[6];
rz(1.5707963267948966) q[6];
x q[1];
x q[6];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
x q[0];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
cx q[6], q[2];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
x q[2];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[5];
cx q[10], q[9];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
h q[5];
x q[4];
x q[9];
rz(1.5707963267948966) q[2];
cx q[9], q[0];
ry(1.5707963267948966) q[2];
cx q[7], q[1];
rz(1.5707963267948966) q[9];
h q[4];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[10];
cx q[6], q[7];
h q[9];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[9];
h q[7];
x q[2];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
x q[6];
cx q[9], q[4];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[1];
x q[2];
ry(1.5707963267948966) q[9];
cx q[2], q[7];
cx q[1], q[2];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
cx q[6], q[2];
rz(1.5707963267948966) q[3];
h q[4];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[7];
h q[0];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
x q[9];
h q[1];
x q[4];
ry(1.5707963267948966) q[5];
x q[5];
x q[5];
h q[9];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[8];
x q[6];
h q[8];
cx q[5], q[10];
h q[1];
h q[4];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
cx q[9], q[7];
h q[7];
h q[3];
h q[5];
x q[5];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[8];
cx q[6], q[1];
h q[8];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
x q[9];
rx(1.5707963267948966) q[7];
cx q[4], q[1];
rz(1.5707963267948966) q[3];
cx q[7], q[9];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[7];
h q[6];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[10];
x q[6];
rz(1.5707963267948966) q[4];
h q[4];
ry(1.5707963267948966) q[0];
cx q[10], q[2];
h q[1];
rz(1.5707963267948966) q[9];
h q[0];
cx q[4], q[2];
h q[7];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[5];
x q[9];
h q[6];
x q[3];
cx q[5], q[3];
ry(1.5707963267948966) q[3];
cx q[2], q[5];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
cx q[3], q[7];
rx(1.5707963267948966) q[4];
cx q[1], q[10];
cx q[5], q[2];
cx q[5], q[2];
cx q[10], q[4];
h q[3];
rx(1.5707963267948966) q[2];
x q[5];
cx q[8], q[9];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
x q[6];
x q[4];
ry(1.5707963267948966) q[4];
h q[9];
rx(1.5707963267948966) q[5];
x q[5];
ry(1.5707963267948966) q[5];
cx q[7], q[0];
cx q[9], q[0];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[7];
h q[0];
h q[8];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
h q[2];
h q[0];
cx q[7], q[6];
h q[8];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
cx q[4], q[0];
ry(1.5707963267948966) q[9];
x q[10];
rz(1.5707963267948966) q[8];
h q[1];
cx q[5], q[6];
rx(1.5707963267948966) q[5];
h q[9];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
cx q[4], q[0];
h q[10];
rx(1.5707963267948966) q[1];
h q[2];
h q[1];
h q[9];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
cx q[6], q[7];
cx q[2], q[10];
h q[3];
x q[6];
rx(1.5707963267948966) q[0];
cx q[9], q[2];
rx(1.5707963267948966) q[3];
cx q[9], q[3];
x q[6];
rx(1.5707963267948966) q[6];
cx q[2], q[9];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[5];
x q[8];
cx q[5], q[2];
rz(1.5707963267948966) q[8];
h q[8];
x q[2];
ry(1.5707963267948966) q[10];
h q[2];
cx q[4], q[7];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
cx q[2], q[10];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[8];
x q[0];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
cx q[7], q[10];
h q[5];
rx(1.5707963267948966) q[7];
h q[9];
x q[5];
ry(1.5707963267948966) q[7];
cx q[10], q[3];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
cx q[3], q[0];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
cx q[2], q[5];
cx q[3], q[5];
ry(1.5707963267948966) q[2];
h q[0];
x q[10];
h q[0];
h q[0];
cx q[2], q[10];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
h q[10];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[10];
h q[1];
rx(1.5707963267948966) q[7];
cx q[8], q[2];
x q[10];
rz(1.5707963267948966) q[9];
x q[2];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
x q[4];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[4];
x q[4];
h q[7];
cx q[6], q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
cx q[1], q[9];
rx(1.5707963267948966) q[8];
h q[1];
rz(1.5707963267948966) q[6];
x q[5];
cx q[10], q[4];
cx q[5], q[10];
rz(1.5707963267948966) q[2];
cx q[1], q[3];
ry(1.5707963267948966) q[3];
h q[1];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
h q[9];
x q[7];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
h q[6];
x q[7];
h q[0];
ry(1.5707963267948966) q[2];
x q[10];
cx q[3], q[10];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
x q[2];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
x q[1];
ry(1.5707963267948966) q[0];
cx q[4], q[3];
ry(1.5707963267948966) q[4];
cx q[9], q[3];
cx q[6], q[0];
x q[2];
rx(1.5707963267948966) q[10];
x q[7];
x q[10];
rx(1.5707963267948966) q[4];
cx q[10], q[0];
cx q[1], q[8];
h q[4];
h q[9];
ry(1.5707963267948966) q[10];
h q[0];
h q[2];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[6];
h q[7];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
h q[4];
h q[1];
h q[0];
h q[6];
rx(1.5707963267948966) q[0];
x q[9];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
cx q[9], q[3];
cx q[9], q[2];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
h q[1];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[0];
cx q[3], q[1];
x q[4];
rx(1.5707963267948966) q[6];
h q[10];
h q[7];
rz(1.5707963267948966) q[4];
cx q[0], q[9];
ry(1.5707963267948966) q[7];
h q[0];
h q[0];
x q[9];
rx(1.5707963267948966) q[3];
cx q[8], q[7];
x q[2];
rx(1.5707963267948966) q[6];
x q[6];
ry(1.5707963267948966) q[7];
x q[9];
cx q[2], q[1];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
x q[10];
cx q[6], q[2];
x q[6];
rz(1.5707963267948966) q[2];
cx q[10], q[9];
cx q[7], q[3];
rz(1.5707963267948966) q[5];
x q[6];
ry(1.5707963267948966) q[9];
x q[10];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[3];
h q[10];
cx q[1], q[3];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[3];
cx q[7], q[5];
rz(1.5707963267948966) q[1];
h q[0];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
h q[0];
h q[9];
x q[10];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
h q[8];
rx(1.5707963267948966) q[5];
h q[1];
cx q[0], q[6];
x q[4];
rx(1.5707963267948966) q[9];
cx q[3], q[4];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
x q[10];
x q[3];
cx q[9], q[6];
x q[3];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[10];
cx q[9], q[1];
rz(1.5707963267948966) q[5];
x q[1];
cx q[4], q[1];
ry(1.5707963267948966) q[8];
cx q[8], q[0];
cx q[6], q[2];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[10];
cx q[4], q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
h q[9];
x q[8];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
cx q[4], q[0];
rz(1.5707963267948966) q[6];
h q[2];
x q[6];
cx q[0], q[9];
rx(1.5707963267948966) q[6];
cx q[3], q[8];
h q[2];
h q[2];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[6];
x q[10];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[7];
cx q[4], q[6];
x q[5];
h q[6];
h q[3];
rz(1.5707963267948966) q[0];
h q[1];
ry(1.5707963267948966) q[3];
cx q[9], q[6];
h q[8];
ry(1.5707963267948966) q[3];
x q[8];
cx q[10], q[4];
h q[4];
cx q[4], q[0];
h q[9];
ry(1.5707963267948966) q[8];
x q[4];
h q[7];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[1];
cx q[7], q[0];
h q[2];
rz(1.5707963267948966) q[10];
cx q[9], q[1];
ry(1.5707963267948966) q[7];
h q[2];
cx q[0], q[5];
x q[10];
rx(1.5707963267948966) q[2];
h q[7];
x q[1];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[8];
h q[4];
h q[5];
ry(1.5707963267948966) q[4];
h q[7];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
h q[1];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
cx q[6], q[2];
cx q[8], q[5];
cx q[5], q[8];
h q[8];
rx(1.5707963267948966) q[6];
cx q[0], q[10];
x q[0];
rz(1.5707963267948966) q[0];
cx q[0], q[4];
ry(1.5707963267948966) q[9];
h q[6];
h q[4];
h q[3];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[7];
cx q[4], q[9];
cx q[3], q[8];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
cx q[4], q[2];
rz(1.5707963267948966) q[3];
x q[9];
rx(1.5707963267948966) q[1];
cx q[8], q[5];
cx q[3], q[8];
cx q[0], q[9];
ry(1.5707963267948966) q[1];
h q[10];
h q[0];
h q[7];
h q[4];
rx(1.5707963267948966) q[3];
cx q[1], q[2];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
cx q[2], q[8];
rz(1.5707963267948966) q[9];
h q[8];
h q[3];
ry(1.5707963267948966) q[8];
cx q[8], q[6];
rx(1.5707963267948966) q[9];
h q[5];
cx q[6], q[3];
x q[1];
cx q[9], q[5];
cx q[9], q[4];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
cx q[8], q[6];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
h q[6];
x q[9];
rx(1.5707963267948966) q[2];
cx q[0], q[7];
x q[8];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[6];
x q[5];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[7];
x q[8];
rz(1.5707963267948966) q[8];
cx q[0], q[8];
cx q[2], q[3];
rz(1.5707963267948966) q[4];
cx q[6], q[5];
x q[8];
h q[8];
h q[4];
ry(1.5707963267948966) q[1];
cx q[0], q[3];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
cx q[8], q[5];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[10];
cx q[2], q[6];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
h q[7];
cx q[8], q[9];
cx q[10], q[7];
x q[4];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
cx q[9], q[1];
h q[9];
h q[5];
x q[10];
rx(1.5707963267948966) q[7];
h q[6];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[1];
x q[7];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[8];
h q[8];
h q[0];
cx q[8], q[4];
cx q[8], q[10];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
h q[9];
h q[5];
rx(1.5707963267948966) q[6];
h q[0];
x q[10];
rz(1.5707963267948966) q[3];
x q[7];
rz(1.5707963267948966) q[2];
cx q[8], q[0];
ry(1.5707963267948966) q[4];
cx q[2], q[0];
x q[3];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
cx q[8], q[10];
x q[1];
rz(1.5707963267948966) q[1];
x q[0];
rz(1.5707963267948966) q[5];
h q[2];
cx q[8], q[1];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[9];
h q[4];
h q[4];
h q[3];
cx q[9], q[6];
rz(1.5707963267948966) q[4];
h q[5];
rx(1.5707963267948966) q[0];
x q[8];
ry(1.5707963267948966) q[2];
cx q[8], q[4];
cx q[2], q[5];
h q[6];
x q[3];
ry(1.5707963267948966) q[3];
x q[0];
h q[5];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[0];
h q[7];
x q[5];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[8];
cx q[9], q[0];
x q[8];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
cx q[7], q[1];
rz(1.5707963267948966) q[9];
cx q[9], q[3];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
cx q[4], q[6];
cx q[6], q[10];
ry(1.5707963267948966) q[2];
x q[10];
rx(1.5707963267948966) q[1];
x q[9];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[10];
cx q[6], q[4];
rx(1.5707963267948966) q[9];
cx q[2], q[7];
h q[5];
cx q[9], q[4];
rx(1.5707963267948966) q[7];
x q[4];
cx q[5], q[7];
cx q[6], q[4];
rz(1.5707963267948966) q[8];
x q[4];
cx q[8], q[4];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[2];
h q[0];
x q[3];
cx q[3], q[7];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
cx q[10], q[0];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
x q[4];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
x q[8];
cx q[4], q[3];
x q[10];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[10];
cx q[8], q[4];
h q[6];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
x q[0];
ry(1.5707963267948966) q[8];
h q[8];
x q[7];
cx q[3], q[10];
h q[5];
cx q[5], q[1];
cx q[3], q[4];
cx q[9], q[0];
x q[7];
cx q[10], q[7];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
h q[1];
cx q[2], q[4];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
x q[0];
cx q[4], q[5];
rz(1.5707963267948966) q[5];
cx q[9], q[10];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
h q[0];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[0];
cx q[6], q[9];
cx q[1], q[10];
h q[0];
cx q[3], q[6];
x q[9];
cx q[2], q[10];
h q[3];
ry(1.5707963267948966) q[10];
x q[1];
rz(1.5707963267948966) q[4];
h q[1];
cx q[1], q[4];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
cx q[1], q[4];
x q[7];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
cx q[5], q[0];
cx q[3], q[9];
cx q[1], q[3];
ry(1.5707963267948966) q[2];
cx q[3], q[1];
cx q[9], q[3];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
x q[1];
h q[6];
cx q[10], q[8];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
cx q[6], q[1];
ry(1.5707963267948966) q[6];
h q[8];
cx q[6], q[0];
x q[9];
h q[8];
x q[10];
x q[1];
rz(1.5707963267948966) q[4];
h q[6];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[10];
h q[7];
rx(1.5707963267948966) q[3];
x q[6];
h q[2];
rz(1.5707963267948966) q[5];
h q[6];
h q[9];
rx(1.5707963267948966) q[0];
cx q[10], q[9];
h q[3];
rx(1.5707963267948966) q[6];
h q[5];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
cx q[10], q[6];
cx q[6], q[8];
cx q[1], q[7];
h q[1];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
cx q[1], q[8];
h q[3];
h q[8];
x q[10];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[0];
h q[3];
x q[3];
cx q[3], q[8];
h q[9];
ry(1.5707963267948966) q[6];
cx q[6], q[3];
h q[5];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[9];
cx q[0], q[8];
cx q[9], q[8];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
h q[2];
x q[3];
h q[5];
rx(1.5707963267948966) q[6];
x q[2];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
h q[1];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[10];
h q[4];
x q[10];
ry(1.5707963267948966) q[0];
x q[2];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
cx q[2], q[10];
rx(1.5707963267948966) q[6];
cx q[9], q[5];
ry(1.5707963267948966) q[10];
cx q[8], q[4];
rx(1.5707963267948966) q[6];
x q[3];
cx q[9], q[3];
x q[2];
x q[5];
ry(1.5707963267948966) q[2];
cx q[10], q[9];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
h q[2];
x q[10];
h q[6];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
h q[1];
ry(1.5707963267948966) q[3];
cx q[0], q[6];
x q[5];
rz(1.5707963267948966) q[6];
x q[8];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
cx q[6], q[3];
h q[4];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[9];
h q[4];
cx q[1], q[8];
ry(1.5707963267948966) q[2];
cx q[8], q[2];
ry(1.5707963267948966) q[3];
h q[0];
x q[8];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
h q[3];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
cx q[2], q[9];
cx q[2], q[1];
rz(1.5707963267948966) q[6];
h q[8];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
cx q[5], q[8];
x q[2];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
x q[2];
ry(1.5707963267948966) q[0];
cx q[1], q[7];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[6];
cx q[3], q[0];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[8];
cx q[1], q[4];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[10];
h q[4];
rx(1.5707963267948966) q[10];
x q[8];
x q[5];
cx q[3], q[1];
cx q[3], q[6];
ry(1.5707963267948966) q[5];
x q[4];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[10];
h q[5];
x q[8];
cx q[2], q[6];
rz(1.5707963267948966) q[5];
x q[5];
x q[3];
cx q[10], q[3];
h q[10];
ry(1.5707963267948966) q[6];
x q[7];
rz(1.5707963267948966) q[2];
x q[7];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
cx q[8], q[3];
h q[1];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[5];
cx q[6], q[0];
rz(1.5707963267948966) q[2];
cx q[10], q[2];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
cx q[3], q[9];
x q[1];
x q[5];
cx q[6], q[10];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[8];
x q[3];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[9];
x q[3];
cx q[9], q[4];
rz(1.5707963267948966) q[10];
cx q[7], q[6];
h q[2];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[10];
cx q[2], q[7];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
x q[7];
ry(1.5707963267948966) q[6];
x q[7];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[4];
x q[3];
rz(1.5707963267948966) q[0];
cx q[6], q[5];
h q[3];
rz(1.5707963267948966) q[10];
x q[7];
rx(1.5707963267948966) q[5];
cx q[0], q[3];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
cx q[5], q[6];
h q[4];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[5];
x q[7];
rx(1.5707963267948966) q[8];
cx q[3], q[5];
ry(1.5707963267948966) q[4];
x q[1];
cx q[5], q[7];
cx q[6], q[7];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
x q[4];
cx q[3], q[9];
cx q[9], q[3];
rz(1.5707963267948966) q[9];
x q[2];
cx q[7], q[9];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[10];
h q[2];
h q[7];
h q[1];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
h q[8];
h q[8];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
h q[2];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
cx q[2], q[9];
x q[7];
cx q[7], q[5];
h q[10];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[3];
h q[9];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[0];
h q[4];
rx(1.5707963267948966) q[1];
cx q[0], q[2];
rz(1.5707963267948966) q[1];
h q[1];
cx q[6], q[10];
rz(1.5707963267948966) q[4];
cx q[2], q[8];
cx q[6], q[5];
rx(1.5707963267948966) q[8];
h q[4];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
cx q[9], q[1];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[4];
x q[1];
cx q[8], q[9];
rz(1.5707963267948966) q[4];
x q[6];
h q[5];
ry(1.5707963267948966) q[8];
x q[3];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
x q[4];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[10];
x q[2];
cx q[3], q[2];
rx(1.5707963267948966) q[6];
h q[6];
cx q[10], q[9];
cx q[6], q[2];
rx(1.5707963267948966) q[9];
h q[5];
h q[1];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
h q[1];
h q[3];
h q[4];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[7];
x q[6];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[7];
cx q[10], q[8];
x q[2];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[0];
h q[4];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
cx q[7], q[4];
cx q[6], q[1];
x q[2];
ry(1.5707963267948966) q[3];
h q[9];
x q[8];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[10];
x q[5];
rx(1.5707963267948966) q[2];
h q[4];
x q[0];
ry(1.5707963267948966) q[10];
h q[4];
rx(1.5707963267948966) q[2];
x q[1];
h q[7];
x q[5];
cx q[10], q[0];
ry(1.5707963267948966) q[0];
x q[5];
rz(1.5707963267948966) q[3];
h q[7];
cx q[10], q[3];
h q[4];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
x q[8];
cx q[5], q[1];
ry(1.5707963267948966) q[5];
x q[1];
h q[7];
rz(1.5707963267948966) q[7];
h q[1];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
h q[3];
rx(1.5707963267948966) q[8];
cx q[3], q[9];
rz(1.5707963267948966) q[3];
cx q[8], q[7];
rx(1.5707963267948966) q[1];
x q[10];
x q[0];
x q[10];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
x q[2];
ry(1.5707963267948966) q[9];
x q[3];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
h q[9];
rz(1.5707963267948966) q[0];
x q[5];
h q[5];
x q[5];
x q[1];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[0];
x q[8];
cx q[5], q[10];
x q[5];
ry(1.5707963267948966) q[8];
x q[4];
x q[7];
h q[9];
h q[3];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[10];
h q[3];
cx q[9], q[2];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[9];
h q[7];
rx(1.5707963267948966) q[5];
h q[4];
x q[2];
rx(1.5707963267948966) q[6];
cx q[6], q[7];
ry(1.5707963267948966) q[7];
x q[5];
cx q[9], q[7];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
cx q[7], q[10];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[9];
cx q[2], q[3];
x q[3];
cx q[9], q[3];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[3];
cx q[7], q[10];
rz(1.5707963267948966) q[7];
cx q[3], q[10];
ry(1.5707963267948966) q[3];
h q[5];
cx q[6], q[0];
cx q[5], q[0];
x q[5];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[9];
cx q[5], q[1];
cx q[1], q[8];
h q[4];
ry(1.5707963267948966) q[4];
cx q[9], q[3];
cx q[8], q[4];
rz(1.5707963267948966) q[7];
h q[4];
cx q[3], q[5];
x q[1];
x q[1];
h q[7];
rz(1.5707963267948966) q[6];
cx q[6], q[4];
rx(1.5707963267948966) q[10];
h q[0];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[7];
cx q[8], q[2];
h q[0];
ry(1.5707963267948966) q[1];
h q[1];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[9];
h q[3];
x q[8];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
cx q[7], q[0];
x q[8];
rz(1.5707963267948966) q[4];
x q[8];
h q[5];
cx q[4], q[9];
x q[0];
x q[3];
ry(1.5707963267948966) q[10];
h q[10];
h q[8];
rz(1.5707963267948966) q[3];
cx q[10], q[9];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
cx q[8], q[10];
cx q[9], q[5];
cx q[1], q[3];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[4];
cx q[9], q[5];
rz(1.5707963267948966) q[10];
cx q[10], q[6];
h q[4];
ry(1.5707963267948966) q[8];
cx q[6], q[8];
rz(1.5707963267948966) q[4];
h q[6];
h q[9];
cx q[6], q[8];
h q[2];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
x q[3];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[10];
x q[6];
x q[7];
rx(1.5707963267948966) q[9];
cx q[7], q[6];
x q[8];
h q[7];
rz(1.5707963267948966) q[8];
x q[8];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
x q[8];
h q[8];
rz(1.5707963267948966) q[2];
x q[6];
h q[9];
ry(1.5707963267948966) q[6];
x q[3];
cx q[4], q[0];
h q[6];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
h q[6];
h q[8];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[10];
h q[8];
h q[7];
rx(1.5707963267948966) q[3];
h q[9];
rx(1.5707963267948966) q[2];
cx q[8], q[6];
ry(1.5707963267948966) q[7];
h q[10];
rz(1.5707963267948966) q[10];
h q[1];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
h q[10];
rx(1.5707963267948966) q[1];
cx q[10], q[3];
x q[8];
rz(1.5707963267948966) q[5];
h q[7];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
cx q[4], q[8];
ry(1.5707963267948966) q[0];
cx q[10], q[9];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
x q[10];
rz(1.5707963267948966) q[4];
x q[0];
h q[0];
rx(1.5707963267948966) q[9];
h q[5];
ry(1.5707963267948966) q[8];
h q[5];
ry(1.5707963267948966) q[10];
h q[4];
cx q[6], q[5];
cx q[10], q[3];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
cx q[7], q[2];
x q[2];
ry(1.5707963267948966) q[9];
x q[7];
cx q[7], q[10];
rz(1.5707963267948966) q[7];
cx q[8], q[4];
x q[6];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
x q[6];
x q[5];
ry(1.5707963267948966) q[4];
h q[4];
x q[4];
x q[0];
ry(1.5707963267948966) q[5];
x q[8];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[7];
cx q[7], q[8];
h q[0];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
x q[10];
rz(1.5707963267948966) q[9];
