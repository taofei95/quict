OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
sx q[17];
ry(1.5707963267948966) q[15];
sw q[10];
fsim(1.5707963267948966, 0) q[11], q[5];
rx(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[4], q[17];
ry(1.5707963267948966) q[18];
sw q[2];
sy q[4];
fsim(1.5707963267948966, 0) q[18], q[9];
ry(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[15], q[6];
sy q[8];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[16];
sw q[1];
sx q[17];
sw q[11];
sy q[13];
sw q[14];
sw q[0];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[13];
sx q[18];
fsim(1.5707963267948966, 0) q[13], q[7];
sx q[18];
ry(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[1], q[5];
sw q[7];
fsim(1.5707963267948966, 0) q[9], q[13];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[13];
sw q[6];
sy q[17];
sx q[16];
ry(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[9], q[17];
fsim(1.5707963267948966, 0) q[6], q[15];
sx q[3];
fsim(1.5707963267948966, 0) q[5], q[6];
rx(1.5707963267948966) q[3];
sx q[9];
fsim(1.5707963267948966, 0) q[6], q[17];
sw q[18];
fsim(1.5707963267948966, 0) q[16], q[14];
fsim(1.5707963267948966, 0) q[5], q[4];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[18], q[5];
fsim(1.5707963267948966, 0) q[3], q[9];
rx(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[18], q[13];
ry(1.5707963267948966) q[1];
sw q[4];
fsim(1.5707963267948966, 0) q[1], q[5];
sw q[10];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
sy q[11];
sy q[1];
fsim(1.5707963267948966, 0) q[10], q[7];
sw q[10];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[12];
sx q[7];
sw q[15];
fsim(1.5707963267948966, 0) q[5], q[7];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[16];
sw q[12];
sy q[0];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[1], q[5];
sx q[16];
fsim(1.5707963267948966, 0) q[17], q[4];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[7], q[0];
ry(1.5707963267948966) q[0];
sx q[1];
sy q[8];
sy q[16];
fsim(1.5707963267948966, 0) q[0], q[2];
fsim(1.5707963267948966, 0) q[3], q[7];
fsim(1.5707963267948966, 0) q[3], q[1];
sw q[4];
sx q[11];
fsim(1.5707963267948966, 0) q[0], q[2];
sx q[11];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[3], q[10];
fsim(1.5707963267948966, 0) q[9], q[16];
sx q[16];
sx q[8];
fsim(1.5707963267948966, 0) q[10], q[7];
sy q[10];
ry(1.5707963267948966) q[7];
sw q[2];
ry(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[13], q[1];
ry(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[11], q[6];
fsim(1.5707963267948966, 0) q[5], q[3];
sy q[17];
fsim(1.5707963267948966, 0) q[14], q[17];
