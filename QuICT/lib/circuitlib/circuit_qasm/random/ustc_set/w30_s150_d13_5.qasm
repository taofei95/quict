OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
ry(1.5707963267948966) q[1];
h q[12];
cx q[11], q[26];
h q[23];
rz(1.5707963267948966) q[1];
h q[2];
h q[21];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[19];
cx q[24], q[6];
h q[10];
ry(1.5707963267948966) q[25];
cx q[23], q[10];
rx(1.5707963267948966) q[27];
cx q[14], q[8];
h q[4];
cx q[20], q[14];
ry(1.5707963267948966) q[3];
h q[19];
ry(1.5707963267948966) q[4];
cx q[8], q[23];
ry(1.5707963267948966) q[25];
x q[4];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[21];
h q[4];
rx(1.5707963267948966) q[10];
x q[2];
rx(1.5707963267948966) q[26];
cx q[18], q[5];
x q[20];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[12];
cx q[23], q[26];
rx(1.5707963267948966) q[11];
x q[2];
cx q[1], q[21];
cx q[17], q[7];
cx q[29], q[3];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[25];
cx q[24], q[6];
rx(1.5707963267948966) q[27];
cx q[26], q[24];
x q[29];
rz(1.5707963267948966) q[27];
x q[16];
cx q[19], q[1];
h q[0];
x q[4];
rx(1.5707963267948966) q[29];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[14];
cx q[12], q[14];
rx(1.5707963267948966) q[3];
h q[26];
rz(1.5707963267948966) q[2];
h q[15];
cx q[21], q[23];
x q[13];
cx q[21], q[11];
rx(1.5707963267948966) q[7];
cx q[15], q[20];
rx(1.5707963267948966) q[25];
h q[3];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[0];
x q[27];
x q[24];
x q[0];
cx q[19], q[20];
x q[0];
rx(1.5707963267948966) q[7];
cx q[29], q[10];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[28];
h q[10];
h q[11];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[10];
h q[18];
rx(1.5707963267948966) q[29];
cx q[5], q[15];
ry(1.5707963267948966) q[17];
x q[16];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[14];
h q[28];
rz(1.5707963267948966) q[27];
ry(1.5707963267948966) q[14];
x q[22];
x q[3];
cx q[9], q[28];
cx q[16], q[9];
cx q[3], q[6];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[18];
h q[2];
rx(1.5707963267948966) q[21];
rz(1.5707963267948966) q[17];
cx q[1], q[26];
cx q[17], q[7];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[15];
x q[16];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[22];
cx q[11], q[4];
cx q[10], q[5];
cx q[20], q[25];
rz(1.5707963267948966) q[26];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[20];
cx q[9], q[23];
ry(1.5707963267948966) q[6];
cx q[23], q[27];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[16];
cx q[22], q[25];
cx q[18], q[12];
cx q[17], q[15];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[9];
x q[27];
rx(1.5707963267948966) q[28];
rx(1.5707963267948966) q[0];
x q[28];
rz(1.5707963267948966) q[6];
x q[18];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[27];
cx q[7], q[18];
