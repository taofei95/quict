OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
rxx(0) q[7], q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
rxx(0) q[7], q[6];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[11];
rxx(0) q[2], q[3];
rz(1.5707963267948966) q[10];
rxx(0) q[11], q[4];
rxx(0) q[7], q[6];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
rxx(0) q[11], q[2];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[1];
rxx(0) q[7], q[6];
rxx(0) q[11], q[9];
rx(1.5707963267948966) q[1];
rxx(0) q[9], q[11];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
rxx(0) q[4], q[0];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
rxx(0) q[5], q[12];
ry(1.5707963267948966) q[1];
rxx(0) q[5], q[2];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
rxx(0) q[0], q[6];
ry(1.5707963267948966) q[4];
rxx(0) q[8], q[2];
ry(1.5707963267948966) q[8];
