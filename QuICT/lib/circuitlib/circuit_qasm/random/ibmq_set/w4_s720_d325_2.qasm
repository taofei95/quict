OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rz(1.5707963267948966) q[0];
x q[0];
sx q[3];
rz(1.5707963267948966) q[2];
x q[3];
cx q[3], q[1];
cx q[3], q[0];
sx q[1];
sx q[2];
cx q[3], q[0];
x q[3];
sx q[3];
sx q[2];
x q[2];
x q[1];
x q[2];
x q[2];
rz(1.5707963267948966) q[1];
sx q[2];
rz(1.5707963267948966) q[1];
cx q[2], q[1];
cx q[3], q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
sx q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
sx q[3];
x q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
sx q[2];
rz(1.5707963267948966) q[0];
x q[2];
rz(1.5707963267948966) q[1];
sx q[2];
x q[1];
cx q[1], q[2];
x q[2];
cx q[3], q[0];
cx q[3], q[1];
sx q[2];
cx q[2], q[0];
x q[1];
cx q[2], q[3];
sx q[2];
cx q[1], q[2];
rz(1.5707963267948966) q[2];
cx q[3], q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
cx q[0], q[3];
cx q[3], q[0];
x q[0];
rz(1.5707963267948966) q[3];
sx q[2];
x q[2];
cx q[1], q[0];
sx q[1];
rz(1.5707963267948966) q[3];
x q[1];
cx q[3], q[1];
cx q[3], q[0];
cx q[2], q[3];
rz(1.5707963267948966) q[0];
x q[0];
x q[0];
rz(1.5707963267948966) q[2];
cx q[0], q[3];
rz(1.5707963267948966) q[0];
cx q[1], q[0];
sx q[3];
cx q[1], q[0];
cx q[0], q[1];
rz(1.5707963267948966) q[0];
x q[3];
rz(1.5707963267948966) q[2];
cx q[2], q[3];
rz(1.5707963267948966) q[3];
sx q[1];
rz(1.5707963267948966) q[1];
cx q[3], q[2];
sx q[2];
sx q[3];
x q[0];
sx q[2];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
cx q[1], q[2];
x q[0];
sx q[1];
sx q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
sx q[2];
rz(1.5707963267948966) q[2];
cx q[1], q[2];
sx q[0];
x q[1];
sx q[3];
x q[2];
sx q[0];
x q[1];
cx q[2], q[0];
cx q[0], q[3];
sx q[3];
rz(1.5707963267948966) q[2];
x q[1];
cx q[3], q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
sx q[0];
cx q[2], q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
cx q[0], q[3];
cx q[2], q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
sx q[0];
sx q[2];
x q[0];
x q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
x q[2];
x q[1];
rz(1.5707963267948966) q[3];
sx q[1];
x q[1];
cx q[1], q[0];
sx q[2];
cx q[1], q[0];
sx q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
cx q[0], q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
cx q[1], q[3];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
x q[3];
x q[0];
x q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
sx q[2];
sx q[1];
cx q[2], q[3];
sx q[0];
rz(1.5707963267948966) q[3];
sx q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
sx q[2];
rz(1.5707963267948966) q[0];
sx q[1];
rz(1.5707963267948966) q[3];
cx q[3], q[2];
x q[0];
rz(1.5707963267948966) q[0];
x q[0];
cx q[3], q[2];
cx q[1], q[3];
sx q[0];
sx q[3];
rz(1.5707963267948966) q[2];
x q[2];
cx q[3], q[1];
x q[3];
sx q[3];
rz(1.5707963267948966) q[2];
cx q[0], q[1];
rz(1.5707963267948966) q[2];
sx q[0];
cx q[3], q[0];
cx q[3], q[2];
cx q[1], q[2];
cx q[2], q[3];
sx q[3];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
sx q[2];
rz(1.5707963267948966) q[1];
x q[3];
rz(1.5707963267948966) q[0];
x q[0];
sx q[2];
cx q[2], q[1];
x q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
sx q[0];
sx q[1];
rz(1.5707963267948966) q[0];
x q[1];
sx q[1];
cx q[0], q[2];
x q[0];
sx q[0];
cx q[3], q[2];
cx q[1], q[2];
rz(1.5707963267948966) q[2];
cx q[3], q[1];
rz(1.5707963267948966) q[1];
x q[2];
cx q[2], q[3];
cx q[1], q[2];
cx q[2], q[3];
sx q[2];
cx q[0], q[1];
x q[2];
sx q[3];
x q[1];
rz(1.5707963267948966) q[2];
sx q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
sx q[3];
cx q[2], q[3];
sx q[0];
cx q[1], q[3];
x q[2];
sx q[0];
cx q[1], q[2];
sx q[2];
x q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
sx q[1];
rz(1.5707963267948966) q[0];
cx q[2], q[0];
rz(1.5707963267948966) q[0];
cx q[2], q[1];
rz(1.5707963267948966) q[3];
cx q[3], q[0];
cx q[3], q[1];
rz(1.5707963267948966) q[1];
sx q[1];
rz(1.5707963267948966) q[2];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
cx q[1], q[2];
x q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
sx q[2];
x q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
sx q[2];
sx q[2];
sx q[0];
sx q[1];
sx q[0];
x q[0];
sx q[1];
sx q[2];
cx q[0], q[3];
rz(1.5707963267948966) q[3];
x q[3];
x q[0];
rz(1.5707963267948966) q[2];
x q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
x q[0];
rz(1.5707963267948966) q[3];
x q[0];
x q[3];
cx q[1], q[3];
sx q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
sx q[1];
rz(1.5707963267948966) q[3];
sx q[3];
cx q[1], q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
cx q[0], q[3];
sx q[3];
sx q[0];
sx q[2];
sx q[0];
sx q[1];
x q[1];
sx q[3];
sx q[2];
rz(1.5707963267948966) q[2];
sx q[1];
sx q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
cx q[0], q[3];
x q[2];
x q[0];
x q[3];
x q[3];
cx q[3], q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
cx q[2], q[1];
cx q[2], q[0];
sx q[1];
cx q[1], q[3];
rz(1.5707963267948966) q[3];
x q[3];
cx q[0], q[1];
x q[0];
rz(1.5707963267948966) q[1];
x q[2];
rz(1.5707963267948966) q[3];
cx q[3], q[2];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
cx q[3], q[0];
sx q[2];
cx q[0], q[1];
x q[0];
sx q[3];
x q[1];
cx q[3], q[0];
x q[0];
x q[0];
sx q[2];
x q[3];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
sx q[0];
sx q[1];
cx q[3], q[2];
cx q[2], q[3];
sx q[0];
sx q[0];
cx q[3], q[1];
rz(1.5707963267948966) q[2];
x q[3];
sx q[1];
x q[0];
sx q[3];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
x q[0];
sx q[1];
x q[0];
rz(1.5707963267948966) q[3];
x q[0];
rz(1.5707963267948966) q[1];
sx q[3];
sx q[0];
x q[0];
rz(1.5707963267948966) q[2];
x q[2];
sx q[1];
cx q[0], q[1];
sx q[0];
x q[3];
x q[1];
sx q[3];
sx q[1];
sx q[2];
cx q[2], q[3];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
cx q[1], q[3];
x q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
cx q[0], q[3];
sx q[0];
rz(1.5707963267948966) q[1];
sx q[2];
rz(1.5707963267948966) q[1];
sx q[3];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
cx q[1], q[3];
cx q[0], q[1];
sx q[3];
sx q[3];
rz(1.5707963267948966) q[0];
x q[3];
x q[1];
sx q[1];
sx q[0];
sx q[3];
sx q[1];
rz(1.5707963267948966) q[1];
sx q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
cx q[1], q[2];
rz(1.5707963267948966) q[0];
sx q[1];
sx q[3];
sx q[3];
x q[2];
x q[1];
rz(1.5707963267948966) q[3];
x q[1];
cx q[3], q[2];
cx q[3], q[2];
sx q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
cx q[2], q[3];
x q[1];
cx q[3], q[1];
x q[2];
x q[1];
rz(1.5707963267948966) q[3];
sx q[3];
cx q[2], q[0];
rz(1.5707963267948966) q[2];
x q[2];
sx q[3];
rz(1.5707963267948966) q[2];
cx q[1], q[3];
cx q[0], q[2];
sx q[0];
x q[1];
cx q[0], q[2];
sx q[2];
rz(1.5707963267948966) q[1];
sx q[3];
x q[2];
cx q[3], q[1];
cx q[1], q[0];
sx q[0];
x q[0];
sx q[0];
x q[2];
cx q[2], q[0];
x q[1];
rz(1.5707963267948966) q[3];
cx q[2], q[1];
rz(1.5707963267948966) q[1];
sx q[1];
cx q[2], q[1];
rz(1.5707963267948966) q[1];
sx q[1];
sx q[0];
sx q[1];
sx q[3];
sx q[3];
sx q[3];
sx q[2];
x q[3];
x q[1];
sx q[0];
x q[1];
rz(1.5707963267948966) q[1];
x q[1];
sx q[2];
sx q[0];
x q[0];
rz(1.5707963267948966) q[2];
x q[3];
cx q[1], q[3];
rz(1.5707963267948966) q[2];
cx q[3], q[1];
rz(1.5707963267948966) q[1];
sx q[2];
rz(1.5707963267948966) q[0];
x q[0];
rz(1.5707963267948966) q[1];
x q[3];
rz(1.5707963267948966) q[3];
x q[0];
cx q[1], q[2];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
sx q[2];
cx q[0], q[2];
cx q[1], q[0];
sx q[1];
x q[1];
cx q[1], q[2];
cx q[0], q[1];
sx q[3];
rz(1.5707963267948966) q[1];
x q[3];
x q[1];
x q[2];
cx q[1], q[0];
rz(1.5707963267948966) q[1];
cx q[2], q[0];
rz(1.5707963267948966) q[1];
sx q[0];
sx q[3];
rz(1.5707963267948966) q[3];
sx q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
cx q[2], q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
x q[1];
rz(1.5707963267948966) q[2];
x q[1];
rz(1.5707963267948966) q[3];
cx q[0], q[2];
cx q[0], q[3];
rz(1.5707963267948966) q[2];
cx q[1], q[0];
cx q[0], q[3];
x q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
sx q[0];
rz(1.5707963267948966) q[3];
cx q[3], q[0];
x q[3];
cx q[3], q[2];
sx q[0];
rz(1.5707963267948966) q[0];
cx q[3], q[1];
rz(1.5707963267948966) q[2];
sx q[2];
sx q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
cx q[3], q[1];
sx q[1];
rz(1.5707963267948966) q[0];
sx q[0];
rz(1.5707963267948966) q[1];
sx q[2];
x q[2];
cx q[0], q[2];
sx q[1];
cx q[1], q[0];
x q[3];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
sx q[0];
x q[3];
x q[2];
cx q[0], q[1];
x q[1];
sx q[0];
rz(1.5707963267948966) q[3];
sx q[1];
x q[0];
cx q[0], q[1];
x q[1];
x q[0];
x q[0];
rz(1.5707963267948966) q[0];
sx q[0];
sx q[1];
x q[2];
x q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
cx q[0], q[3];
x q[2];
sx q[2];
cx q[0], q[3];
cx q[2], q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
sx q[1];
cx q[1], q[3];
sx q[2];
cx q[0], q[3];
rz(1.5707963267948966) q[3];
x q[1];
cx q[3], q[2];
rz(1.5707963267948966) q[2];
cx q[3], q[0];
x q[2];
x q[0];
cx q[3], q[0];
rz(1.5707963267948966) q[1];
x q[3];
rz(1.5707963267948966) q[3];
x q[1];
x q[1];
rz(1.5707963267948966) q[3];
cx q[1], q[2];
cx q[0], q[2];
rz(1.5707963267948966) q[0];
sx q[3];
sx q[2];
sx q[0];
x q[3];
cx q[0], q[2];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
sx q[2];
x q[1];
x q[2];
x q[1];
sx q[0];
x q[3];
rz(1.5707963267948966) q[0];
sx q[2];
x q[3];
rz(1.5707963267948966) q[2];
sx q[2];
x q[3];
x q[1];
rz(1.5707963267948966) q[2];
cx q[2], q[1];
sx q[0];
sx q[0];
sx q[2];
x q[3];
sx q[1];
x q[2];
sx q[2];
cx q[1], q[2];
x q[1];
sx q[3];
cx q[2], q[1];
cx q[1], q[0];
cx q[2], q[1];
sx q[1];
x q[1];
x q[2];
rz(1.5707963267948966) q[1];
sx q[0];
cx q[0], q[1];
x q[0];
x q[1];
x q[2];
rz(1.5707963267948966) q[2];
x q[1];
x q[0];
cx q[2], q[1];
x q[0];
sx q[0];
cx q[0], q[1];
cx q[3], q[2];
cx q[2], q[0];
rz(1.5707963267948966) q[2];
sx q[2];
sx q[1];
rz(1.5707963267948966) q[0];
cx q[3], q[1];
rz(1.5707963267948966) q[2];
x q[0];
sx q[0];
x q[0];
cx q[3], q[2];
rz(1.5707963267948966) q[0];
cx q[0], q[2];
cx q[2], q[3];
rz(1.5707963267948966) q[0];
x q[0];
x q[2];
rz(1.5707963267948966) q[2];
cx q[3], q[0];
sx q[2];
rz(1.5707963267948966) q[1];
x q[0];
sx q[1];
x q[0];
sx q[3];
rz(1.5707963267948966) q[1];
x q[2];
sx q[3];
x q[3];
cx q[3], q[0];
x q[0];
sx q[0];
rz(1.5707963267948966) q[1];
cx q[3], q[2];
rz(1.5707963267948966) q[1];
sx q[0];
rz(1.5707963267948966) q[0];
cx q[2], q[0];
cx q[1], q[3];
sx q[1];
x q[1];
x q[2];
sx q[2];
cx q[1], q[2];
x q[0];
sx q[1];
x q[0];
sx q[1];
sx q[2];
sx q[2];
x q[3];
sx q[1];
rz(1.5707963267948966) q[0];
sx q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[3];
x q[0];
sx q[1];
rz(1.5707963267948966) q[0];
x q[0];
sx q[3];
x q[2];
sx q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
cx q[1], q[3];
cx q[3], q[1];
rz(1.5707963267948966) q[2];
cx q[0], q[2];
x q[1];
rz(1.5707963267948966) q[0];
x q[2];
rz(1.5707963267948966) q[0];
