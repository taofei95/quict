OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
rz(1.5707963267948966) q[12];
h q[10];
h q[9];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[15];
ry(1.5707963267948966) q[11];
h q[14];
h q[10];
cx q[5], q[2];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[4];
cx q[10], q[11];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[6];
cx q[15], q[0];
h q[12];
cx q[9], q[4];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[10];
h q[12];
h q[12];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[14];
h q[13];
cx q[2], q[5];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[2];
h q[6];
h q[1];
rz(1.5707963267948966) q[15];
ry(1.5707963267948966) q[6];
h q[5];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[2];
h q[6];
cx q[11], q[0];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[10];
cx q[14], q[11];
cx q[8], q[7];
cx q[5], q[12];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[7];
cx q[15], q[9];
h q[10];
rz(1.5707963267948966) q[6];
h q[2];
h q[3];
cx q[2], q[12];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[13];
cx q[10], q[15];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[10];
cx q[2], q[3];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[0];
h q[12];
