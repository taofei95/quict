OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[1];
cx q[1], q[0];
x q[0];
x q[0];
sx q[0];
sx q[0];
sx q[1];
rz(1.5707963267948966) q[1];
cx q[0], q[1];
sx q[0];
sx q[0];
sx q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
cx q[1], q[0];
rz(1.5707963267948966) q[0];
sx q[0];
cx q[0], q[1];
sx q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
x q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
cx q[0], q[1];
x q[0];
x q[1];
x q[0];
sx q[0];
cx q[1], q[0];
sx q[1];
rz(1.5707963267948966) q[0];
x q[0];
cx q[1], q[0];
sx q[1];
x q[0];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
x q[1];
rz(1.5707963267948966) q[0];
sx q[0];
rz(1.5707963267948966) q[1];
x q[1];
x q[0];
x q[0];
rz(1.5707963267948966) q[1];
x q[0];
cx q[1], q[0];
sx q[0];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
sx q[0];
sx q[0];
x q[1];
cx q[1], q[0];
x q[0];
cx q[1], q[0];
x q[0];
rz(1.5707963267948966) q[0];
x q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
sx q[1];
cx q[1], q[0];
x q[1];
cx q[0], q[1];
x q[1];
x q[0];
x q[1];
cx q[0], q[1];
x q[1];
cx q[1], q[0];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
cx q[1], q[0];
x q[1];
sx q[1];
rz(1.5707963267948966) q[0];
sx q[0];
rz(1.5707963267948966) q[0];
x q[1];
x q[0];
sx q[1];
x q[0];
cx q[1], q[0];
x q[0];
sx q[0];
rz(1.5707963267948966) q[1];
x q[0];
sx q[0];
x q[0];
sx q[0];
rz(1.5707963267948966) q[1];
sx q[0];
x q[1];
sx q[0];
sx q[0];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
sx q[0];
sx q[1];
cx q[1], q[0];
x q[1];
sx q[1];
rz(1.5707963267948966) q[1];
sx q[1];
sx q[1];
rz(1.5707963267948966) q[1];
sx q[1];
x q[0];
x q[1];
sx q[0];
sx q[0];
rz(1.5707963267948966) q[0];
sx q[0];
cx q[1], q[0];
cx q[0], q[1];
rz(1.5707963267948966) q[0];
x q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
x q[1];
rz(1.5707963267948966) q[0];
sx q[0];
sx q[0];
sx q[0];
x q[1];
x q[1];
x q[0];
sx q[1];
x q[1];
x q[0];
x q[1];
rz(1.5707963267948966) q[1];
cx q[1], q[0];
rz(1.5707963267948966) q[0];
sx q[0];
cx q[0], q[1];
sx q[0];
cx q[0], q[1];
x q[1];
cx q[1], q[0];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
x q[1];
sx q[0];
x q[0];
sx q[0];
x q[1];
cx q[1], q[0];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
sx q[0];
x q[1];
sx q[0];
x q[1];
cx q[1], q[0];
sx q[1];
x q[1];
rz(1.5707963267948966) q[0];
sx q[0];
cx q[0], q[1];
sx q[0];
sx q[0];
cx q[0], q[1];
sx q[1];
x q[1];
rz(1.5707963267948966) q[0];
x q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
sx q[1];
cx q[1], q[0];
x q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
x q[0];
cx q[0], q[1];
x q[0];
x q[0];
sx q[1];
sx q[0];
cx q[0], q[1];
cx q[1], q[0];
rz(1.5707963267948966) q[1];
sx q[0];
x q[0];
x q[0];
sx q[0];
rz(1.5707963267948966) q[1];
x q[1];
x q[0];
rz(1.5707963267948966) q[0];
cx q[1], q[0];
sx q[0];
sx q[1];
rz(1.5707963267948966) q[1];
x q[1];
rz(1.5707963267948966) q[0];
x q[0];
x q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
cx q[1], q[0];
cx q[1], q[0];
rz(1.5707963267948966) q[1];
sx q[0];
sx q[0];
sx q[1];
cx q[1], q[0];
cx q[1], q[0];
rz(1.5707963267948966) q[0];
sx q[1];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
cx q[1], q[0];
sx q[0];
cx q[1], q[0];
cx q[0], q[1];
cx q[0], q[1];
sx q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
cx q[0], q[1];
x q[1];
rz(1.5707963267948966) q[1];
cx q[0], q[1];
sx q[0];
cx q[0], q[1];
cx q[0], q[1];
cx q[1], q[0];
x q[1];
cx q[1], q[0];
sx q[1];
x q[0];
rz(1.5707963267948966) q[0];
cx q[1], q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
sx q[0];
rz(1.5707963267948966) q[1];
x q[0];
cx q[0], q[1];
cx q[0], q[1];
cx q[1], q[0];
x q[1];
rz(1.5707963267948966) q[0];
x q[1];
x q[0];
x q[0];
x q[0];
sx q[0];
sx q[1];
rz(1.5707963267948966) q[0];
x q[1];
rz(1.5707963267948966) q[0];
sx q[0];
x q[1];
rz(1.5707963267948966) q[1];
x q[0];
x q[0];
rz(1.5707963267948966) q[1];
x q[0];
rz(1.5707963267948966) q[0];
sx q[0];
rz(1.5707963267948966) q[1];
sx q[0];
x q[0];
x q[1];
cx q[1], q[0];
rz(1.5707963267948966) q[1];
x q[0];
cx q[0], q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[0];
sx q[1];
cx q[0], q[1];
x q[1];
cx q[0], q[1];
sx q[1];
cx q[1], q[0];
x q[0];
sx q[0];
sx q[0];
x q[1];
x q[1];
cx q[1], q[0];
x q[1];
rz(1.5707963267948966) q[0];
cx q[1], q[0];
sx q[1];
sx q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
x q[0];
rz(1.5707963267948966) q[0];
x q[1];
cx q[0], q[1];
rz(1.5707963267948966) q[0];
x q[0];
cx q[1], q[0];
sx q[1];
sx q[1];
rz(1.5707963267948966) q[0];
cx q[1], q[0];
x q[0];
sx q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
sx q[0];
sx q[1];
rz(1.5707963267948966) q[1];
x q[0];
sx q[1];
sx q[0];
rz(1.5707963267948966) q[0];
sx q[1];
sx q[0];
rz(1.5707963267948966) q[0];
sx q[1];
sx q[0];
rz(1.5707963267948966) q[1];
x q[0];
x q[0];
x q[1];
sx q[1];
x q[0];
cx q[1], q[0];
rz(1.5707963267948966) q[0];
x q[1];
sx q[0];
sx q[0];
sx q[0];
cx q[0], q[1];
x q[1];
sx q[1];
sx q[0];
sx q[0];
cx q[1], q[0];
cx q[0], q[1];
cx q[0], q[1];
sx q[0];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
x q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
sx q[0];
cx q[0], q[1];
rz(1.5707963267948966) q[0];
x q[0];
rz(1.5707963267948966) q[1];
sx q[1];
sx q[1];
sx q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
x q[0];
x q[1];
rz(1.5707963267948966) q[1];
cx q[0], q[1];
x q[1];
x q[1];
sx q[0];
x q[1];
x q[1];
sx q[1];
sx q[1];
sx q[0];
x q[1];
x q[0];
x q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
x q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
sx q[1];
rz(1.5707963267948966) q[0];
sx q[1];
sx q[1];
rz(1.5707963267948966) q[0];
x q[0];
sx q[0];
sx q[0];
x q[0];
rz(1.5707963267948966) q[0];
sx q[0];
cx q[1], q[0];
sx q[1];
rz(1.5707963267948966) q[0];
