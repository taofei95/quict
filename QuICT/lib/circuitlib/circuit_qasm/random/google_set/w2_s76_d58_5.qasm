OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
sw q[0];
fsim(1.5707963267948966, 0) q[1], q[0];
sx q[0];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[1], q[0];
sy q[1];
sy q[0];
sy q[0];
fsim(1.5707963267948966, 0) q[0], q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[0], q[1];
sy q[1];
rx(1.5707963267948966) q[0];
sx q[0];
sx q[0];
rx(1.5707963267948966) q[1];
sx q[1];
fsim(1.5707963267948966, 0) q[0], q[1];
sy q[0];
ry(1.5707963267948966) q[1];
sy q[0];
sx q[1];
ry(1.5707963267948966) q[1];
sy q[1];
sw q[1];
ry(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[0], q[1];
sw q[1];
fsim(1.5707963267948966, 0) q[1], q[0];
ry(1.5707963267948966) q[1];
sx q[0];
sw q[0];
rx(1.5707963267948966) q[1];
sw q[0];
sw q[1];
sw q[0];
fsim(1.5707963267948966, 0) q[0], q[1];
fsim(1.5707963267948966, 0) q[0], q[1];
sy q[0];
fsim(1.5707963267948966, 0) q[0], q[1];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[0], q[1];
sw q[0];
sw q[1];
ry(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[0], q[1];
sy q[1];
fsim(1.5707963267948966, 0) q[0], q[1];
sy q[0];
sw q[1];
rx(1.5707963267948966) q[0];
sw q[0];
fsim(1.5707963267948966, 0) q[1], q[0];
sx q[1];
fsim(1.5707963267948966, 0) q[1], q[0];
sy q[0];
sx q[1];
sw q[1];
sw q[0];
sx q[1];
sy q[0];
sw q[0];
fsim(1.5707963267948966, 0) q[0], q[1];
sx q[0];
fsim(1.5707963267948966, 0) q[1], q[0];
rx(1.5707963267948966) q[0];
sy q[0];
fsim(1.5707963267948966, 0) q[0], q[1];
sx q[1];
ry(1.5707963267948966) q[1];
