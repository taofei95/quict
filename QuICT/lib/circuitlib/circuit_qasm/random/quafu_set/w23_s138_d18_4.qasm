OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
cx q[12], q[13];
ry(1.5707963267948966) q[21];
h q[1];
ry(1.5707963267948966) q[1];
cx q[12], q[16];
cx q[7], q[11];
rz(1.5707963267948966) q[21];
cx q[20], q[14];
cx q[21], q[7];
ry(1.5707963267948966) q[15];
cx q[18], q[15];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[7];
h q[5];
rx(1.5707963267948966) q[17];
cx q[4], q[10];
ry(1.5707963267948966) q[10];
cx q[4], q[20];
h q[8];
cx q[18], q[14];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[22];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[15];
cx q[20], q[22];
ry(1.5707963267948966) q[2];
cx q[6], q[5];
cx q[10], q[8];
ry(1.5707963267948966) q[5];
h q[11];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[8];
cx q[5], q[2];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[14];
h q[15];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[17];
cx q[16], q[20];
h q[7];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[8];
cx q[7], q[18];
rz(1.5707963267948966) q[3];
cx q[8], q[1];
h q[0];
h q[11];
rx(1.5707963267948966) q[20];
h q[2];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[16];
cx q[14], q[5];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[22];
rx(1.5707963267948966) q[6];
cx q[18], q[22];
ry(1.5707963267948966) q[19];
h q[18];
rz(1.5707963267948966) q[22];
cx q[6], q[12];
ry(1.5707963267948966) q[20];
h q[3];
h q[20];
rz(1.5707963267948966) q[14];
h q[2];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[1];
cx q[14], q[10];
h q[12];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[17];
cx q[21], q[17];
h q[18];
ry(1.5707963267948966) q[3];
cx q[20], q[8];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[22];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[15];
cx q[9], q[13];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[20];
cx q[3], q[5];
rx(1.5707963267948966) q[17];
cx q[6], q[2];
h q[0];
cx q[20], q[14];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[21];
h q[13];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[22];
h q[12];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[9];
h q[5];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[15];
h q[2];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[20];
cx q[6], q[16];
rz(1.5707963267948966) q[13];
cx q[12], q[20];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[2];
h q[20];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[22];
ry(1.5707963267948966) q[2];
h q[6];
ry(1.5707963267948966) q[16];
h q[20];
h q[9];
h q[13];
ry(1.5707963267948966) q[0];
h q[20];
rz(1.5707963267948966) q[15];
cx q[7], q[22];
h q[0];
