OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
rx(1.5707963267948966) q[4];
h q[1];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[9];
h q[21];
rx(1.5707963267948966) q[16];
h q[7];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[21];
rx(1.5707963267948966) q[1];
h q[7];
cx q[21], q[17];
rz(1.5707963267948966) q[3];
h q[9];
h q[13];
cx q[15], q[17];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
h q[14];
cx q[7], q[13];
h q[21];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[22];
h q[16];
ry(1.5707963267948966) q[15];
h q[4];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[18];
h q[21];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[18];
cx q[14], q[4];
rz(1.5707963267948966) q[14];
cx q[8], q[15];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[1];
cx q[23], q[9];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[15];
h q[15];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[21];
cx q[1], q[12];
cx q[20], q[16];
rz(1.5707963267948966) q[2];
cx q[15], q[20];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[4];
h q[22];
rx(1.5707963267948966) q[0];
h q[22];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
cx q[15], q[2];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[22];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[5];
h q[6];
cx q[19], q[9];
cx q[18], q[11];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[22];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[5];
cx q[11], q[22];
ry(1.5707963267948966) q[1];
h q[17];
cx q[4], q[15];
h q[9];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
h q[21];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[15];
h q[7];
cx q[22], q[14];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[7];
cx q[13], q[21];
h q[23];
rx(1.5707963267948966) q[16];
cx q[10], q[23];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[12];
cx q[13], q[4];
cx q[3], q[23];
rx(1.5707963267948966) q[16];
h q[13];
ry(1.5707963267948966) q[22];
h q[21];
cx q[23], q[20];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[3];
h q[19];
cx q[12], q[18];
h q[12];
cx q[20], q[23];
ry(1.5707963267948966) q[6];
cx q[19], q[12];
h q[17];
rz(1.5707963267948966) q[16];
