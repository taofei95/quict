OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
cx q[12], q[18];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[4];
cx q[8], q[3];
h q[1];
h q[17];
x q[7];
x q[2];
rz(1.5707963267948966) q[12];
h q[1];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[16];
cx q[10], q[9];
rx(1.5707963267948966) q[0];
cx q[18], q[3];
x q[19];
h q[15];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[8];
h q[2];
h q[6];
h q[16];
rx(1.5707963267948966) q[2];
h q[5];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[5];
cx q[11], q[19];
cx q[10], q[1];
cx q[1], q[10];
cx q[1], q[15];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
h q[7];
cx q[2], q[11];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[3];
x q[6];
x q[6];
cx q[1], q[3];
cx q[16], q[0];
rz(1.5707963267948966) q[18];
cx q[10], q[3];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[15];
cx q[7], q[19];
cx q[4], q[8];
x q[16];
rz(1.5707963267948966) q[15];
cx q[15], q[4];
h q[4];
rz(1.5707963267948966) q[10];
h q[8];
h q[16];
ry(1.5707963267948966) q[19];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[12];
h q[17];
ry(1.5707963267948966) q[9];
cx q[8], q[14];
rx(1.5707963267948966) q[19];
x q[0];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[14];
h q[10];
x q[15];
x q[17];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[14];
cx q[5], q[18];
cx q[11], q[3];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[17];
x q[9];
x q[17];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[12];
cx q[1], q[18];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[14];
h q[13];
cx q[18], q[7];
