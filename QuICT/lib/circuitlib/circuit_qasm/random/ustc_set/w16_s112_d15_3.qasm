OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
cx q[6], q[3];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
h q[9];
h q[5];
x q[2];
ry(1.5707963267948966) q[5];
cx q[14], q[1];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[9];
h q[4];
rz(1.5707963267948966) q[2];
h q[11];
h q[1];
x q[10];
x q[11];
rz(1.5707963267948966) q[5];
x q[12];
h q[10];
h q[15];
h q[12];
cx q[6], q[11];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[3];
cx q[8], q[14];
x q[10];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[2];
h q[8];
cx q[7], q[8];
ry(1.5707963267948966) q[5];
cx q[8], q[13];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
x q[13];
rx(1.5707963267948966) q[3];
cx q[8], q[7];
x q[0];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[2];
cx q[15], q[0];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
cx q[8], q[12];
cx q[15], q[11];
rx(1.5707963267948966) q[14];
h q[14];
x q[9];
x q[13];
x q[1];
rz(1.5707963267948966) q[5];
x q[3];
cx q[4], q[6];
h q[14];
x q[12];
rx(1.5707963267948966) q[4];
cx q[5], q[10];
cx q[12], q[6];
cx q[12], q[1];
x q[10];
h q[15];
ry(1.5707963267948966) q[1];
x q[10];
cx q[11], q[5];
x q[11];
h q[11];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
x q[0];
h q[1];
h q[5];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[12];
cx q[2], q[13];
rx(1.5707963267948966) q[12];
x q[6];
cx q[9], q[8];
rz(1.5707963267948966) q[0];
x q[3];
x q[6];
h q[8];
cx q[6], q[8];
rz(1.5707963267948966) q[3];
h q[12];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[8];
x q[0];
x q[2];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[10];
cx q[7], q[14];
rx(1.5707963267948966) q[6];
cx q[15], q[1];
cx q[5], q[7];
rx(1.5707963267948966) q[2];
h q[12];
x q[3];
x q[2];
h q[6];
x q[10];
cx q[14], q[11];
rz(1.5707963267948966) q[8];
h q[7];
