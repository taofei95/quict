OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
x q[0];
h q[10];
x q[18];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[3];
h q[18];
cx q[7], q[19];
x q[3];
rx(1.5707963267948966) q[10];
cx q[0], q[11];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[10];
h q[9];
x q[1];
cx q[10], q[18];
rz(1.5707963267948966) q[19];
h q[15];
cx q[11], q[16];
rz(1.5707963267948966) q[10];
x q[5];
cx q[18], q[5];
cx q[1], q[19];
cx q[0], q[7];
cx q[9], q[15];
h q[9];
cx q[8], q[15];
h q[14];
rx(1.5707963267948966) q[11];
cx q[10], q[3];
x q[7];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[15];
cx q[18], q[2];
rz(1.5707963267948966) q[3];
x q[1];
cx q[7], q[19];
h q[13];
x q[15];
h q[15];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[13];
cx q[3], q[15];
ry(1.5707963267948966) q[15];
cx q[5], q[7];
cx q[16], q[11];
cx q[8], q[11];
rx(1.5707963267948966) q[19];
h q[7];
rx(1.5707963267948966) q[1];
cx q[3], q[10];
rz(1.5707963267948966) q[10];
h q[4];
x q[4];
cx q[5], q[11];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
cx q[14], q[11];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[19];
x q[3];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[15];
cx q[7], q[1];
x q[16];
rz(1.5707963267948966) q[19];
x q[18];
cx q[11], q[4];
rx(1.5707963267948966) q[5];
x q[0];
x q[15];
h q[12];
x q[14];
x q[9];
cx q[18], q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
x q[3];
h q[15];
x q[11];
ry(1.5707963267948966) q[1];
h q[16];
h q[6];
cx q[15], q[18];
h q[16];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[17];
x q[9];
x q[4];
cx q[6], q[10];
rx(1.5707963267948966) q[11];
cx q[18], q[2];
cx q[3], q[10];
h q[3];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[4];
h q[8];
cx q[11], q[3];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
x q[5];
cx q[16], q[4];
x q[9];
x q[9];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[9];
cx q[14], q[15];
x q[8];
cx q[6], q[14];
cx q[13], q[16];
cx q[16], q[0];
h q[10];
