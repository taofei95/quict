OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
h q[6];
rx(1.5707963267948966) q[1];
cx q[9], q[10];
x q[9];
rz(1.5707963267948966) q[1];
cx q[4], q[9];
cx q[8], q[9];
cx q[6], q[11];
cx q[2], q[0];
cx q[10], q[4];
x q[11];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[12];
x q[12];
cx q[10], q[8];
h q[6];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
h q[0];
x q[12];
x q[11];
h q[1];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
x q[8];
ry(1.5707963267948966) q[11];
cx q[2], q[6];
x q[7];
rx(1.5707963267948966) q[11];
x q[7];
h q[1];
h q[12];
ry(1.5707963267948966) q[2];
h q[1];
h q[11];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[2];
h q[11];
rx(1.5707963267948966) q[1];
h q[5];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
h q[2];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
h q[12];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[3];
cx q[9], q[4];
cx q[4], q[7];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[5];
cx q[7], q[4];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[12];
h q[9];
x q[0];
h q[2];
x q[2];
h q[1];
cx q[3], q[1];
ry(1.5707963267948966) q[1];
h q[4];
ry(1.5707963267948966) q[5];
cx q[11], q[0];
cx q[8], q[11];
x q[2];
x q[12];
rz(1.5707963267948966) q[1];
cx q[9], q[2];
ry(1.5707963267948966) q[5];
x q[9];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
h q[9];
