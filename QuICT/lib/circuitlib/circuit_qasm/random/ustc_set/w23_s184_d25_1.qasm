OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
rz(1.5707963267948966) q[16];
x q[8];
cx q[6], q[3];
h q[3];
rz(1.5707963267948966) q[13];
cx q[7], q[17];
rz(1.5707963267948966) q[21];
ry(1.5707963267948966) q[2];
h q[0];
h q[8];
ry(1.5707963267948966) q[17];
x q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[14];
cx q[17], q[18];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[13];
h q[2];
h q[3];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[2];
x q[21];
x q[10];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[2];
cx q[18], q[14];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[22];
rz(1.5707963267948966) q[16];
cx q[22], q[8];
h q[1];
rz(1.5707963267948966) q[11];
cx q[13], q[0];
h q[17];
rx(1.5707963267948966) q[3];
cx q[5], q[20];
cx q[5], q[13];
rz(1.5707963267948966) q[11];
h q[3];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[15];
x q[20];
cx q[21], q[18];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[18];
cx q[5], q[10];
cx q[15], q[21];
x q[3];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[14];
x q[16];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[13];
h q[4];
h q[18];
rx(1.5707963267948966) q[20];
x q[13];
rz(1.5707963267948966) q[16];
x q[18];
cx q[15], q[9];
cx q[16], q[1];
x q[17];
x q[8];
cx q[3], q[9];
h q[18];
cx q[21], q[2];
h q[13];
rz(1.5707963267948966) q[22];
cx q[18], q[6];
rz(1.5707963267948966) q[14];
cx q[14], q[12];
cx q[18], q[8];
cx q[14], q[2];
cx q[16], q[2];
h q[10];
rx(1.5707963267948966) q[5];
cx q[21], q[0];
h q[3];
ry(1.5707963267948966) q[22];
rz(1.5707963267948966) q[13];
h q[17];
ry(1.5707963267948966) q[0];
h q[21];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[14];
h q[10];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[10];
cx q[16], q[6];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[8];
h q[17];
cx q[15], q[17];
ry(1.5707963267948966) q[13];
x q[6];
h q[14];
h q[0];
cx q[9], q[8];
cx q[1], q[10];
rx(1.5707963267948966) q[8];
h q[7];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[6];
cx q[22], q[12];
x q[9];
x q[15];
rz(1.5707963267948966) q[16];
cx q[22], q[3];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[18];
cx q[0], q[15];
rz(1.5707963267948966) q[13];
cx q[19], q[6];
rx(1.5707963267948966) q[14];
x q[11];
x q[16];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[3];
cx q[11], q[22];
rz(1.5707963267948966) q[18];
h q[1];
h q[10];
h q[14];
cx q[20], q[22];
h q[19];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[10];
h q[12];
x q[16];
cx q[5], q[3];
x q[10];
x q[9];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[16];
cx q[0], q[19];
cx q[11], q[19];
cx q[11], q[1];
cx q[12], q[0];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[9];
cx q[14], q[13];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[6];
x q[10];
cx q[8], q[3];
ry(1.5707963267948966) q[10];
h q[10];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[14];
x q[9];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[14];
cx q[8], q[11];
x q[8];
cx q[1], q[4];
cx q[2], q[15];
x q[19];
cx q[16], q[2];
h q[11];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[22];
h q[7];
cx q[11], q[14];
x q[16];
