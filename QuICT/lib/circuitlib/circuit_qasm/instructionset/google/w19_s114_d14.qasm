OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
sx q[16];
sw q[8];
fsim(1.5707963267948966, 0) q[1], q[4];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[5];
sw q[14];
fsim(1.5707963267948966, 0) q[15], q[1];
sw q[14];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[14];
sy q[13];
sy q[3];
sx q[10];
sy q[18];
fsim(1.5707963267948966, 0) q[4], q[8];
sy q[14];
fsim(1.5707963267948966, 0) q[15], q[4];
rx(1.5707963267948966) q[18];
sy q[1];
fsim(1.5707963267948966, 0) q[9], q[1];
fsim(1.5707963267948966, 0) q[2], q[18];
sw q[16];
rx(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[7], q[1];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[15];
fsim(1.5707963267948966, 0) q[18], q[1];
rx(1.5707963267948966) q[4];
sy q[14];
sw q[1];
ry(1.5707963267948966) q[10];
sy q[8];
rx(1.5707963267948966) q[1];
sx q[6];
sw q[3];
ry(1.5707963267948966) q[17];
sy q[8];
fsim(1.5707963267948966, 0) q[11], q[6];
sy q[6];
ry(1.5707963267948966) q[11];
sx q[17];
rx(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[6], q[12];
sw q[14];
sx q[18];
sy q[11];
ry(1.5707963267948966) q[10];
sw q[18];
sy q[12];
sx q[15];
sw q[3];
sx q[4];
fsim(1.5707963267948966, 0) q[5], q[15];
sy q[16];
sx q[10];
sw q[5];
sw q[3];
sx q[14];
ry(1.5707963267948966) q[10];
sx q[8];
rx(1.5707963267948966) q[2];
sy q[13];
fsim(1.5707963267948966, 0) q[4], q[7];
rx(1.5707963267948966) q[4];
sy q[16];
sy q[15];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[14];
fsim(1.5707963267948966, 0) q[8], q[15];
sy q[3];
fsim(1.5707963267948966, 0) q[2], q[0];
sw q[15];
sx q[5];
rx(1.5707963267948966) q[14];
sx q[0];
sy q[14];
sy q[10];
fsim(1.5707963267948966, 0) q[0], q[8];
sy q[6];
ry(1.5707963267948966) q[0];
sw q[3];
sw q[15];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[13], q[4];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[15], q[16];
sw q[15];
sx q[5];
sx q[9];
rx(1.5707963267948966) q[12];
sw q[10];
rx(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[7], q[2];
fsim(1.5707963267948966, 0) q[18], q[5];
sx q[9];
sy q[17];
rx(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[8], q[0];
sw q[9];
sx q[10];
fsim(1.5707963267948966, 0) q[5], q[12];
sy q[3];
fsim(1.5707963267948966, 0) q[1], q[18];
fsim(1.5707963267948966, 0) q[12], q[13];
sy q[9];
ry(1.5707963267948966) q[15];
sy q[12];
fsim(1.5707963267948966, 0) q[2], q[7];
sx q[18];
sx q[2];
fsim(1.5707963267948966, 0) q[9], q[12];
sy q[2];
