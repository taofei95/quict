OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
x q[4];
cx q[0], q[9];
cx q[5], q[9];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[11];
x q[6];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[7];
cx q[5], q[8];
rx(1.5707963267948966) q[2];
x q[8];
x q[10];
rz(1.5707963267948966) q[12];
h q[10];
x q[8];
cx q[10], q[4];
cx q[0], q[6];
rx(1.5707963267948966) q[8];
cx q[7], q[11];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
cx q[1], q[6];
h q[9];
ry(1.5707963267948966) q[12];
h q[11];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[12];
cx q[10], q[8];
ry(1.5707963267948966) q[12];
h q[6];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[8];
cx q[7], q[12];
ry(1.5707963267948966) q[7];
h q[6];
ry(1.5707963267948966) q[8];
x q[9];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[2];
h q[12];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
x q[3];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[5];
h q[5];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[7];
h q[9];
rz(1.5707963267948966) q[12];
h q[5];
rz(1.5707963267948966) q[5];
cx q[3], q[10];
h q[6];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
h q[1];
ry(1.5707963267948966) q[8];
x q[9];
h q[11];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
h q[10];
cx q[7], q[4];
x q[5];
h q[3];
h q[9];
rx(1.5707963267948966) q[2];
x q[0];
rx(1.5707963267948966) q[5];
cx q[6], q[8];
