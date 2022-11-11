OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
rz(1.5707963267948966) q[9];
cx q[0], q[10];
x q[18];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[25];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[4];
cx q[18], q[11];
rz(1.5707963267948966) q[12];
cx q[5], q[12];
rx(1.5707963267948966) q[7];
cx q[24], q[1];
x q[27];
cx q[24], q[2];
rx(1.5707963267948966) q[24];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[1];
cx q[22], q[10];
ry(1.5707963267948966) q[13];
h q[0];
ry(1.5707963267948966) q[28];
cx q[2], q[22];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[11];
cx q[19], q[12];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[17];
h q[25];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[21];
cx q[27], q[29];
ry(1.5707963267948966) q[26];
cx q[28], q[15];
x q[14];
h q[13];
x q[8];
rz(1.5707963267948966) q[29];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[17];
h q[26];
ry(1.5707963267948966) q[10];
x q[16];
x q[6];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[9];
cx q[25], q[13];
ry(1.5707963267948966) q[23];
cx q[24], q[16];
rx(1.5707963267948966) q[20];
x q[16];
rz(1.5707963267948966) q[10];
h q[10];
cx q[6], q[14];
h q[5];
rx(1.5707963267948966) q[18];
cx q[29], q[7];
cx q[4], q[18];
rx(1.5707963267948966) q[11];
cx q[27], q[18];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[28];
cx q[13], q[2];
rz(1.5707963267948966) q[12];
x q[8];
x q[6];
ry(1.5707963267948966) q[23];
h q[26];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[3];
h q[11];
h q[22];
h q[0];
h q[13];
cx q[18], q[7];
x q[17];
cx q[25], q[6];
ry(1.5707963267948966) q[9];
h q[1];
h q[22];
x q[7];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[24];
cx q[12], q[0];
rz(1.5707963267948966) q[26];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[17];
cx q[23], q[1];
h q[17];
rx(1.5707963267948966) q[16];
cx q[5], q[28];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[10];
cx q[7], q[5];
x q[28];
rx(1.5707963267948966) q[19];
cx q[9], q[24];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[7];
cx q[5], q[16];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
x q[5];
h q[7];
h q[4];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[3];
h q[0];
ry(1.5707963267948966) q[11];
cx q[2], q[7];
cx q[14], q[3];
cx q[21], q[25];
cx q[20], q[7];
rz(1.5707963267948966) q[28];
ry(1.5707963267948966) q[2];
h q[5];
x q[29];
rz(1.5707963267948966) q[24];
cx q[11], q[16];
ry(1.5707963267948966) q[26];
x q[21];
cx q[16], q[19];
ry(1.5707963267948966) q[13];
x q[25];
ry(1.5707963267948966) q[23];
cx q[23], q[25];
x q[25];
x q[24];
cx q[15], q[2];
cx q[7], q[24];
x q[16];
cx q[18], q[27];
rx(1.5707963267948966) q[22];
cx q[0], q[28];
ry(1.5707963267948966) q[3];
h q[16];
cx q[7], q[17];
x q[22];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[23];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[0];
h q[28];
rx(1.5707963267948966) q[24];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[22];
x q[13];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[10];
cx q[25], q[9];
ry(1.5707963267948966) q[28];
rz(1.5707963267948966) q[15];
x q[26];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
cx q[14], q[27];
x q[22];
rx(1.5707963267948966) q[21];
x q[4];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[24];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[28];
rz(1.5707963267948966) q[14];
x q[26];
x q[18];
cx q[27], q[5];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[20];
h q[9];
cx q[15], q[6];
ry(1.5707963267948966) q[19];
cx q[4], q[13];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[21];
h q[14];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[23];
rx(1.5707963267948966) q[29];
h q[19];
cx q[18], q[16];
cx q[13], q[14];
cx q[13], q[20];
x q[16];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
x q[24];
x q[13];
x q[9];
x q[24];
ry(1.5707963267948966) q[8];
cx q[3], q[2];
rx(1.5707963267948966) q[16];
x q[29];
x q[18];
cx q[16], q[26];
rx(1.5707963267948966) q[26];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[18];
h q[27];
ry(1.5707963267948966) q[25];
