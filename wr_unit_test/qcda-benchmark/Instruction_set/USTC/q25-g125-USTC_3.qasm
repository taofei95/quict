OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[14];
cx q[7], q[23];
x q[19];
cx q[23], q[5];
ry(1.5707963267948966) q[15];
cx q[2], q[19];
x q[23];
x q[9];
rx(1.5707963267948966) q[24];
x q[21];
ry(1.5707963267948966) q[5];
cx q[14], q[13];
cx q[18], q[8];
x q[15];
cx q[20], q[18];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[10];
x q[17];
cx q[11], q[10];
x q[22];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[20];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
cx q[7], q[9];
h q[14];
rz(1.5707963267948966) q[14];
x q[20];
cx q[4], q[14];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[21];
x q[22];
rz(1.5707963267948966) q[24];
x q[1];
cx q[5], q[16];
cx q[17], q[4];
rx(1.5707963267948966) q[20];
cx q[18], q[3];
cx q[12], q[6];
cx q[19], q[11];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[9];
cx q[21], q[13];
rz(1.5707963267948966) q[4];
cx q[22], q[19];
x q[23];
rx(1.5707963267948966) q[4];
h q[14];
rx(1.5707963267948966) q[0];
h q[16];
h q[7];
rx(1.5707963267948966) q[24];
x q[4];
ry(1.5707963267948966) q[6];
h q[18];
cx q[14], q[22];
cx q[2], q[4];
cx q[10], q[22];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[10];
h q[5];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[22];
h q[3];
rx(1.5707963267948966) q[24];
rx(1.5707963267948966) q[14];
x q[1];
h q[7];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[24];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[23];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[24];
h q[11];
x q[19];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[4];
h q[16];
h q[18];
rz(1.5707963267948966) q[4];
x q[3];
rz(1.5707963267948966) q[7];
h q[21];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[13];
cx q[9], q[23];
x q[3];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[21];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[24];
cx q[17], q[20];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[2];
x q[18];
cx q[15], q[5];
h q[19];
x q[2];
ry(1.5707963267948966) q[16];
h q[16];
cx q[12], q[13];
h q[3];
x q[22];
ry(1.5707963267948966) q[18];
x q[15];
ry(1.5707963267948966) q[14];
h q[23];
rz(1.5707963267948966) q[5];
h q[16];
x q[11];
