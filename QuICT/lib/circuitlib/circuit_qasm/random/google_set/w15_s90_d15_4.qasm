OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
sx q[14];
sy q[1];
sx q[14];
sw q[13];
sw q[8];
sx q[2];
sy q[13];
sw q[5];
sy q[12];
fsim(1.5707963267948966, 0) q[11], q[5];
sx q[11];
sx q[8];
sy q[14];
sx q[6];
sy q[1];
sw q[10];
sy q[14];
ry(1.5707963267948966) q[4];
sx q[12];
sw q[14];
sy q[8];
sy q[11];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
sy q[12];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[0], q[8];
fsim(1.5707963267948966, 0) q[3], q[14];
sx q[6];
rx(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[10], q[5];
sw q[10];
fsim(1.5707963267948966, 0) q[14], q[13];
ry(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[3], q[5];
sy q[6];
sw q[6];
ry(1.5707963267948966) q[13];
sx q[5];
fsim(1.5707963267948966, 0) q[8], q[1];
rx(1.5707963267948966) q[7];
sx q[9];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[10];
sx q[14];
sy q[6];
sw q[4];
sy q[13];
rx(1.5707963267948966) q[3];
sy q[13];
ry(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[10], q[8];
fsim(1.5707963267948966, 0) q[9], q[13];
ry(1.5707963267948966) q[10];
sx q[1];
sw q[2];
sw q[12];
sy q[14];
rx(1.5707963267948966) q[8];
sw q[6];
sx q[3];
sy q[10];
sw q[1];
fsim(1.5707963267948966, 0) q[11], q[2];
sw q[2];
sw q[6];
sx q[5];
sy q[13];
sw q[1];
rx(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[10], q[6];
ry(1.5707963267948966) q[9];
sy q[11];
ry(1.5707963267948966) q[7];
sw q[2];
fsim(1.5707963267948966, 0) q[13], q[1];
sw q[4];
rx(1.5707963267948966) q[9];
sw q[14];
ry(1.5707963267948966) q[8];
sw q[13];
sx q[13];
sw q[11];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[11], q[7];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[6];
sy q[12];
sy q[11];
