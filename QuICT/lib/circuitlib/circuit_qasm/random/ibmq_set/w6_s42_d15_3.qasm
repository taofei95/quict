OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
sx q[5];
x q[4];
rz(1.5707963267948966) q[3];
x q[1];
rz(1.5707963267948966) q[3];
x q[5];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
sx q[3];
cx q[0], q[1];
sx q[3];
x q[4];
x q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
cx q[1], q[2];
x q[3];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
x q[0];
rz(1.5707963267948966) q[1];
cx q[1], q[4];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
sx q[3];
sx q[0];
cx q[1], q[2];
x q[4];
rz(1.5707963267948966) q[2];
x q[1];
x q[4];
cx q[4], q[2];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
x q[4];
x q[4];
rz(1.5707963267948966) q[5];
x q[3];
x q[4];
sx q[3];
x q[0];
rz(1.5707963267948966) q[2];
