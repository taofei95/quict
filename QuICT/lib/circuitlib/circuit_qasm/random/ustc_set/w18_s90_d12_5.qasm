OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[9];
h q[6];
ry(1.5707963267948966) q[3];
cx q[4], q[9];
cx q[8], q[4];
rz(1.5707963267948966) q[6];
cx q[6], q[15];
x q[10];
x q[7];
ry(1.5707963267948966) q[14];
cx q[3], q[13];
h q[2];
cx q[10], q[7];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
cx q[4], q[14];
h q[7];
rx(1.5707963267948966) q[16];
h q[9];
x q[15];
h q[11];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[3];
h q[14];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[14];
cx q[15], q[13];
x q[16];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[10];
h q[12];
x q[1];
ry(1.5707963267948966) q[15];
cx q[9], q[12];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
h q[7];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[8];
x q[17];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[14];
h q[9];
cx q[12], q[3];
cx q[7], q[12];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
h q[7];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[13];
h q[2];
ry(1.5707963267948966) q[4];
h q[17];
x q[10];
rz(1.5707963267948966) q[16];
cx q[13], q[3];
ry(1.5707963267948966) q[6];
cx q[1], q[16];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[6];
h q[15];
cx q[0], q[15];
cx q[17], q[13];
ry(1.5707963267948966) q[7];
x q[0];
h q[17];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[7];
x q[14];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
h q[3];
rx(1.5707963267948966) q[12];
h q[0];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[11];
h q[15];
cx q[17], q[6];
rx(1.5707963267948966) q[16];
h q[11];
cx q[15], q[14];
