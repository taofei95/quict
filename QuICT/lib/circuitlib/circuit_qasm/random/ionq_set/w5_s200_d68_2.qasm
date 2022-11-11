OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
rxx(0) q[1], q[3];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
rxx(0) q[1], q[0];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
rxx(0) q[3], q[4];
rz(1.5707963267948966) q[2];
rxx(0) q[0], q[1];
rx(1.5707963267948966) q[0];
rxx(0) q[1], q[4];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rxx(0) q[2], q[3];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rxx(0) q[2], q[4];
rxx(0) q[3], q[0];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
rxx(0) q[1], q[4];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
rxx(0) q[1], q[2];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
rxx(0) q[0], q[1];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rxx(0) q[4], q[2];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rxx(0) q[3], q[0];
ry(1.5707963267948966) q[1];
rxx(0) q[4], q[0];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
rxx(0) q[0], q[1];
rxx(0) q[0], q[1];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
rxx(0) q[2], q[3];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
rxx(0) q[0], q[4];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
rxx(0) q[0], q[1];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
rxx(0) q[2], q[4];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
rxx(0) q[2], q[1];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[3];
rxx(0) q[0], q[3];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
rxx(0) q[2], q[4];
rx(1.5707963267948966) q[0];
rxx(0) q[2], q[1];
rxx(0) q[3], q[2];
rxx(0) q[2], q[4];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
rxx(0) q[3], q[2];
rxx(0) q[2], q[3];
rz(1.5707963267948966) q[2];
rxx(0) q[0], q[3];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
rxx(0) q[3], q[4];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
rxx(0) q[3], q[4];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
rxx(0) q[0], q[1];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
rxx(0) q[0], q[3];
rx(1.5707963267948966) q[0];
rxx(0) q[2], q[3];
rx(1.5707963267948966) q[2];
rxx(0) q[1], q[2];
rz(1.5707963267948966) q[1];
rxx(0) q[1], q[3];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
