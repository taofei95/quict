OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
rxx(0) q[1], q[0];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
rxx(0) q[1], q[0];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
rxx(0) q[0], q[1];
rz(1.5707963267948966) q[0];
rxx(0) q[1], q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
