OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
sw q[5];
sx q[5];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
sw q[4];
sx q[6];
sw q[1];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[4];
sy q[2];
sx q[1];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[6];
sx q[5];
ry(1.5707963267948966) q[3];
sw q[4];
rx(1.5707963267948966) q[2];
sw q[6];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
sw q[0];
sx q[4];
fsim(1.5707963267948966, 0) q[6], q[0];
fsim(1.5707963267948966, 0) q[6], q[0];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
sw q[5];
fsim(1.5707963267948966, 0) q[3], q[2];
ry(1.5707963267948966) q[1];