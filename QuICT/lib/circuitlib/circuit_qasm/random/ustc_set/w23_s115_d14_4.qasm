OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
x q[6];
x q[20];
ry(1.5707963267948966) q[8];
x q[0];
rz(1.5707963267948966) q[16];
cx q[5], q[17];
cx q[19], q[16];
cx q[5], q[16];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[1];
x q[0];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[10];
x q[15];
h q[4];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[7];
x q[21];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[0];
h q[9];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[3];
cx q[0], q[6];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[11];
h q[6];
x q[22];
rz(1.5707963267948966) q[18];
cx q[18], q[7];
ry(1.5707963267948966) q[3];
cx q[3], q[10];
cx q[5], q[17];
x q[4];
ry(1.5707963267948966) q[4];
x q[22];
rx(1.5707963267948966) q[7];
x q[0];
x q[17];
h q[12];
rz(1.5707963267948966) q[19];
cx q[12], q[9];
h q[17];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[5];
h q[1];
rz(1.5707963267948966) q[8];
h q[15];
x q[12];
h q[2];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[20];
h q[20];
cx q[22], q[5];
rx(1.5707963267948966) q[22];
cx q[13], q[17];
cx q[11], q[9];
rz(1.5707963267948966) q[13];
h q[18];
h q[20];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[17];
ry(1.5707963267948966) q[19];
rx(1.5707963267948966) q[10];
x q[16];
cx q[6], q[15];
rx(1.5707963267948966) q[18];
x q[7];
x q[16];
x q[21];
h q[5];
cx q[10], q[21];
x q[21];
x q[0];
cx q[21], q[13];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[19];
cx q[15], q[3];
h q[15];
x q[18];
cx q[19], q[16];
h q[16];
cx q[2], q[16];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[21];
ry(1.5707963267948966) q[22];
cx q[22], q[21];
rz(1.5707963267948966) q[7];
x q[14];
x q[2];
h q[1];
rx(1.5707963267948966) q[3];
h q[0];
cx q[5], q[15];
cx q[15], q[3];
cx q[3], q[13];
cx q[12], q[6];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[20];
x q[8];
h q[15];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[11];
cx q[2], q[7];
ry(1.5707963267948966) q[3];
