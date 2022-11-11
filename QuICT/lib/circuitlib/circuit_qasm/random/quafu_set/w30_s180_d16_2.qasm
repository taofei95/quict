OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
h q[28];
rz(1.5707963267948966) q[22];
h q[6];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[14];
h q[7];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[29];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[3];
h q[15];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[5];
cx q[15], q[27];
cx q[0], q[26];
h q[25];
ry(1.5707963267948966) q[9];
cx q[17], q[27];
ry(1.5707963267948966) q[7];
cx q[1], q[25];
h q[13];
cx q[24], q[29];
cx q[18], q[0];
ry(1.5707963267948966) q[6];
cx q[25], q[3];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[27];
cx q[9], q[8];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[20];
cx q[7], q[4];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[16];
h q[13];
rz(1.5707963267948966) q[23];
rx(1.5707963267948966) q[24];
h q[1];
rx(1.5707963267948966) q[26];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[13];
cx q[20], q[11];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[6];
cx q[28], q[5];
cx q[9], q[27];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
cx q[6], q[17];
cx q[24], q[23];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[9];
cx q[8], q[2];
rx(1.5707963267948966) q[17];
h q[21];
rz(1.5707963267948966) q[10];
cx q[27], q[10];
rx(1.5707963267948966) q[17];
h q[12];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[26];
cx q[4], q[7];
h q[29];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[29];
rx(1.5707963267948966) q[24];
ry(1.5707963267948966) q[20];
rz(1.5707963267948966) q[25];
rx(1.5707963267948966) q[24];
rx(1.5707963267948966) q[24];
cx q[29], q[23];
rx(1.5707963267948966) q[25];
cx q[21], q[12];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[21];
cx q[14], q[20];
h q[27];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
h q[22];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[20];
h q[29];
ry(1.5707963267948966) q[29];
cx q[29], q[23];
rz(1.5707963267948966) q[29];
cx q[2], q[19];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[6];
h q[17];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[25];
cx q[2], q[16];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[20];
h q[0];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[15];
h q[24];
rz(1.5707963267948966) q[28];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[11];
cx q[13], q[3];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[13];
h q[25];
cx q[26], q[1];
cx q[16], q[13];
cx q[5], q[0];
h q[18];
rz(1.5707963267948966) q[21];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[27];
h q[29];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[7];
h q[20];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[10];
cx q[25], q[29];
h q[12];
h q[19];
cx q[9], q[20];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
cx q[18], q[25];
rz(1.5707963267948966) q[29];
cx q[24], q[28];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
cx q[18], q[2];
rz(1.5707963267948966) q[29];
h q[13];
cx q[8], q[29];
ry(1.5707963267948966) q[6];
h q[22];
rx(1.5707963267948966) q[25];
cx q[15], q[18];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[8];
cx q[19], q[5];
rx(1.5707963267948966) q[4];
h q[11];
rx(1.5707963267948966) q[0];
h q[28];
