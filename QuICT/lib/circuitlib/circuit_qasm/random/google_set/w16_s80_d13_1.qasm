OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
sy q[13];
sw q[13];
sw q[6];
rx(1.5707963267948966) q[8];
sw q[6];
fsim(1.5707963267948966, 0) q[14], q[0];
sw q[1];
sy q[15];
sy q[11];
sx q[14];
fsim(1.5707963267948966, 0) q[9], q[15];
sw q[10];
ry(1.5707963267948966) q[0];
sw q[2];
ry(1.5707963267948966) q[8];
sw q[6];
fsim(1.5707963267948966, 0) q[8], q[7];
fsim(1.5707963267948966, 0) q[12], q[8];
sx q[0];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[15];
sw q[0];
sw q[13];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[8];
sy q[6];
rx(1.5707963267948966) q[9];
sx q[5];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[3], q[9];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[8];
sy q[15];
sy q[11];
rx(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[6], q[8];
fsim(1.5707963267948966, 0) q[10], q[7];
sw q[8];
fsim(1.5707963267948966, 0) q[2], q[10];
rx(1.5707963267948966) q[8];
sw q[1];
sw q[10];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[9];
sy q[1];
ry(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[11], q[15];
rx(1.5707963267948966) q[7];
fsim(1.5707963267948966, 0) q[0], q[1];
sw q[3];
fsim(1.5707963267948966, 0) q[5], q[12];
sy q[6];
ry(1.5707963267948966) q[6];
sy q[13];
sw q[7];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
sy q[0];
sw q[5];
fsim(1.5707963267948966, 0) q[8], q[0];
sy q[3];
sw q[4];
ry(1.5707963267948966) q[8];
sy q[3];
ry(1.5707963267948966) q[0];
sw q[6];
sx q[9];
sy q[13];
fsim(1.5707963267948966, 0) q[5], q[15];
sx q[14];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[12], q[7];
sy q[8];
fsim(1.5707963267948966, 0) q[6], q[14];
rx(1.5707963267948966) q[2];
sw q[10];
