OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
cx q[26], q[19];
rx(1.5707963267948966) q[13];
x q[26];
cx q[3], q[21];
cx q[22], q[20];
rx(1.5707963267948966) q[9];
cx q[14], q[24];
rx(1.5707963267948966) q[13];
h q[24];
rz(1.5707963267948966) q[9];
cx q[15], q[8];
cx q[27], q[10];
rz(1.5707963267948966) q[21];
h q[25];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[24];
rx(1.5707963267948966) q[3];
cx q[11], q[12];
x q[5];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[13];
x q[20];
h q[2];
x q[9];
cx q[6], q[7];
x q[16];
cx q[0], q[23];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[7];
h q[9];
x q[22];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[22];
x q[11];
rx(1.5707963267948966) q[11];
cx q[7], q[12];
ry(1.5707963267948966) q[11];
x q[10];
x q[4];
h q[2];
x q[8];
h q[6];
cx q[5], q[15];
rx(1.5707963267948966) q[6];
h q[4];
rz(1.5707963267948966) q[25];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[26];
x q[21];
h q[5];
rz(1.5707963267948966) q[23];
x q[23];
rz(1.5707963267948966) q[10];
cx q[15], q[11];
cx q[10], q[8];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[12];
x q[8];
h q[8];
cx q[23], q[19];
x q[27];
h q[10];
rx(1.5707963267948966) q[19];
cx q[5], q[26];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[19];
h q[15];
x q[8];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[23];
ry(1.5707963267948966) q[27];
cx q[3], q[6];
h q[4];
ry(1.5707963267948966) q[13];
cx q[9], q[11];
rx(1.5707963267948966) q[25];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[24];
x q[12];
rx(1.5707963267948966) q[11];
cx q[14], q[17];
cx q[14], q[23];
rx(1.5707963267948966) q[8];
cx q[27], q[21];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
cx q[8], q[24];
rz(1.5707963267948966) q[21];
x q[21];
rz(1.5707963267948966) q[8];
cx q[7], q[18];
cx q[16], q[15];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[10];
x q[20];
x q[18];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[23];
x q[4];
cx q[21], q[3];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[26];
x q[13];
rx(1.5707963267948966) q[3];
cx q[21], q[7];
rz(1.5707963267948966) q[22];
x q[7];
x q[11];
cx q[11], q[3];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[20];
h q[8];
x q[8];
x q[2];
ry(1.5707963267948966) q[26];
cx q[6], q[2];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[21];
rz(1.5707963267948966) q[6];
x q[23];
x q[8];
x q[0];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
h q[13];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[14];
h q[19];
cx q[20], q[18];
h q[19];
ry(1.5707963267948966) q[8];
x q[19];
cx q[28], q[10];
rx(1.5707963267948966) q[6];
h q[5];
cx q[10], q[6];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[13];
cx q[16], q[23];
rx(1.5707963267948966) q[17];
x q[24];
cx q[8], q[1];
cx q[14], q[6];
h q[23];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[17];
x q[22];
rz(1.5707963267948966) q[26];
x q[18];
cx q[16], q[0];
h q[2];
x q[3];
h q[27];
rz(1.5707963267948966) q[24];
h q[16];
rx(1.5707963267948966) q[13];
x q[6];
cx q[8], q[5];
cx q[23], q[19];
cx q[25], q[14];
h q[20];
cx q[14], q[21];
cx q[21], q[5];
