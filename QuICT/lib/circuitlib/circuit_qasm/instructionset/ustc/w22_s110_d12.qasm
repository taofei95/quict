OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
h q[17];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[4];
x q[16];
h q[3];
rz(1.5707963267948966) q[10];
h q[7];
x q[20];
cx q[21], q[6];
x q[18];
h q[6];
rz(1.5707963267948966) q[7];
cx q[5], q[17];
cx q[13], q[0];
ry(1.5707963267948966) q[3];
x q[11];
cx q[14], q[19];
ry(1.5707963267948966) q[2];
cx q[6], q[14];
cx q[6], q[14];
x q[17];
ry(1.5707963267948966) q[10];
x q[16];
cx q[21], q[4];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[18];
h q[17];
cx q[19], q[4];
h q[0];
ry(1.5707963267948966) q[6];
cx q[5], q[15];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[8];
h q[10];
rx(1.5707963267948966) q[15];
cx q[19], q[4];
cx q[16], q[9];
ry(1.5707963267948966) q[10];
x q[14];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
cx q[11], q[6];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[19];
cx q[0], q[7];
rx(1.5707963267948966) q[20];
cx q[18], q[19];
cx q[9], q[4];
cx q[11], q[1];
x q[21];
h q[16];
rz(1.5707963267948966) q[4];
x q[10];
x q[15];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[20];
x q[17];
h q[2];
rz(1.5707963267948966) q[15];
cx q[11], q[8];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[4];
cx q[15], q[10];
x q[3];
cx q[10], q[15];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
cx q[1], q[14];
rx(1.5707963267948966) q[12];
cx q[1], q[18];
ry(1.5707963267948966) q[3];
h q[21];
x q[1];
rz(1.5707963267948966) q[8];
cx q[10], q[21];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
cx q[5], q[19];
h q[7];
cx q[16], q[13];
ry(1.5707963267948966) q[2];
x q[3];
rz(1.5707963267948966) q[19];
x q[8];
rz(1.5707963267948966) q[19];
cx q[21], q[2];
rz(1.5707963267948966) q[21];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[18];
x q[4];
ry(1.5707963267948966) q[9];
cx q[3], q[17];
cx q[7], q[21];
x q[4];
x q[0];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[11];
x q[13];
cx q[16], q[0];
x q[6];
x q[9];
cx q[20], q[5];
rx(1.5707963267948966) q[9];
