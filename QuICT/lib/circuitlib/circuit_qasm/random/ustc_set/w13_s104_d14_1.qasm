OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
x q[3];
h q[3];
x q[10];
x q[10];
x q[5];
x q[10];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[1];
cx q[12], q[0];
x q[5];
rx(1.5707963267948966) q[11];
cx q[12], q[6];
rx(1.5707963267948966) q[11];
h q[1];
ry(1.5707963267948966) q[3];
x q[3];
h q[4];
ry(1.5707963267948966) q[4];
x q[12];
x q[7];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
h q[6];
cx q[0], q[1];
h q[2];
x q[1];
rx(1.5707963267948966) q[9];
x q[11];
rx(1.5707963267948966) q[2];
x q[1];
rz(1.5707963267948966) q[1];
x q[5];
ry(1.5707963267948966) q[9];
x q[11];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
x q[9];
x q[11];
h q[11];
rz(1.5707963267948966) q[10];
h q[12];
ry(1.5707963267948966) q[12];
cx q[8], q[7];
cx q[9], q[8];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
x q[0];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
x q[7];
h q[9];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
h q[8];
x q[4];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
cx q[8], q[6];
h q[12];
ry(1.5707963267948966) q[10];
h q[4];
rz(1.5707963267948966) q[3];
h q[9];
h q[3];
h q[5];
rx(1.5707963267948966) q[7];
cx q[5], q[7];
rx(1.5707963267948966) q[0];
x q[10];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[11];
cx q[7], q[11];
rx(1.5707963267948966) q[8];
cx q[3], q[1];
h q[6];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[8];
x q[12];
ry(1.5707963267948966) q[5];
h q[10];
cx q[7], q[2];
cx q[8], q[1];
h q[6];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[12];
h q[12];
cx q[12], q[11];
cx q[3], q[10];
ry(1.5707963267948966) q[0];
x q[12];
cx q[12], q[9];
