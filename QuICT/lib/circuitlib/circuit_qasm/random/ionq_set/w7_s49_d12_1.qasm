OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
rxx(0) q[4], q[0];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
rxx(0) q[2], q[0];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
rxx(0) q[1], q[6];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rxx(0) q[0], q[4];
rxx(0) q[6], q[1];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
rxx(0) q[6], q[2];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
rxx(0) q[1], q[6];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
rxx(0) q[5], q[4];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
