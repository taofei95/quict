OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
creg c[26];
rz(1.5707963267948966) q[23];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
h q[20];
cx q[12], q[3];
rz(1.5707963267948966) q[3];
cx q[12], q[19];
ry(1.5707963267948966) q[22];
cx q[10], q[9];
x q[20];
h q[14];
rz(1.5707963267948966) q[14];
cx q[3], q[8];
h q[0];
cx q[24], q[22];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[5];
cx q[3], q[6];
cx q[10], q[1];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[21];
cx q[19], q[23];
h q[6];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[19];
x q[22];
h q[6];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[23];
h q[10];
rz(1.5707963267948966) q[20];
x q[18];
cx q[2], q[0];
cx q[16], q[4];
rz(1.5707963267948966) q[22];
h q[10];
rx(1.5707963267948966) q[16];
x q[17];
h q[15];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[2];
h q[1];
x q[15];
rz(1.5707963267948966) q[16];
x q[13];
ry(1.5707963267948966) q[13];
h q[24];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[20];
x q[25];
h q[21];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[18];
x q[14];
ry(1.5707963267948966) q[12];
cx q[25], q[22];
h q[5];
rx(1.5707963267948966) q[3];
x q[22];
cx q[3], q[15];
h q[9];
rz(1.5707963267948966) q[5];
cx q[0], q[11];
x q[2];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
h q[5];
cx q[8], q[16];
cx q[18], q[21];
ry(1.5707963267948966) q[2];
h q[11];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[4];
h q[7];
x q[16];
rx(1.5707963267948966) q[12];
cx q[14], q[9];
ry(1.5707963267948966) q[1];
x q[21];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[17];
x q[4];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[19];
h q[25];
cx q[10], q[0];
cx q[24], q[14];
rz(1.5707963267948966) q[15];
cx q[14], q[4];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[17];
h q[11];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[14];
cx q[21], q[25];
rx(1.5707963267948966) q[11];
cx q[25], q[7];
ry(1.5707963267948966) q[25];
cx q[21], q[13];
x q[10];
rx(1.5707963267948966) q[4];
x q[4];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[9];
x q[21];
x q[12];
rz(1.5707963267948966) q[15];
cx q[15], q[21];
x q[2];
h q[6];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[2];
h q[13];
x q[9];
h q[6];
rz(1.5707963267948966) q[11];
cx q[6], q[25];
rx(1.5707963267948966) q[5];
