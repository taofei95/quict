OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
sw q[1];
fsim(1.5707963267948966, 0) q[19], q[9];
fsim(1.5707963267948966, 0) q[3], q[7];
ry(1.5707963267948966) q[0];
sw q[1];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[15], q[11];
sw q[4];
sx q[13];
sx q[12];
sx q[1];
sw q[6];
sx q[0];
sw q[17];
ry(1.5707963267948966) q[19];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[1];
sx q[7];
fsim(1.5707963267948966, 0) q[8], q[12];
rx(1.5707963267948966) q[12];
sx q[12];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[1];
sx q[18];
fsim(1.5707963267948966, 0) q[17], q[18];
rx(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[3], q[18];
sx q[12];
fsim(1.5707963267948966, 0) q[19], q[4];
ry(1.5707963267948966) q[18];
sx q[17];
sw q[4];
rx(1.5707963267948966) q[17];
sy q[7];
rx(1.5707963267948966) q[2];
sw q[13];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[2], q[1];
fsim(1.5707963267948966, 0) q[3], q[4];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[9];
sw q[0];
ry(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[5], q[6];
ry(1.5707963267948966) q[14];
fsim(1.5707963267948966, 0) q[14], q[10];
sx q[6];
sw q[19];
sx q[8];
sw q[18];
sy q[5];
rx(1.5707963267948966) q[3];
sx q[16];
ry(1.5707963267948966) q[10];
sx q[3];
fsim(1.5707963267948966, 0) q[12], q[10];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
sw q[14];
sx q[15];
sy q[19];
sx q[4];
fsim(1.5707963267948966, 0) q[7], q[11];
rx(1.5707963267948966) q[2];
sw q[10];
sx q[9];
sw q[0];
sw q[8];
rx(1.5707963267948966) q[13];
sw q[16];
ry(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[13], q[7];
fsim(1.5707963267948966, 0) q[6], q[15];
sy q[4];
fsim(1.5707963267948966, 0) q[9], q[7];
sw q[7];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[17];
sy q[15];
sx q[0];
ry(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[9], q[12];
sx q[16];
sw q[13];
fsim(1.5707963267948966, 0) q[19], q[13];
ry(1.5707963267948966) q[19];
sx q[5];
rx(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[15], q[19];
ry(1.5707963267948966) q[13];
sx q[16];
rx(1.5707963267948966) q[19];
sy q[2];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[13];
