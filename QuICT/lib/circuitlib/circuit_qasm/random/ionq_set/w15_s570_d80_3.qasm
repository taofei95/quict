OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[2];
rxx(0) q[3], q[1];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[4];
rxx(0) q[2], q[10];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[7];
rxx(0) q[1], q[13];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[11];
rxx(0) q[9], q[0];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[14];
rxx(0) q[5], q[4];
rz(1.5707963267948966) q[4];
rxx(0) q[11], q[12];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rxx(0) q[10], q[11];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[12];
rxx(0) q[0], q[5];
rxx(0) q[4], q[13];
rxx(0) q[7], q[14];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
rxx(0) q[6], q[14];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[11];
rxx(0) q[0], q[2];
rz(1.5707963267948966) q[12];
rxx(0) q[14], q[10];
rxx(0) q[7], q[3];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[13];
rxx(0) q[4], q[8];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[12];
rxx(0) q[14], q[6];
rxx(0) q[7], q[11];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[11];
rxx(0) q[12], q[1];
ry(1.5707963267948966) q[13];
rxx(0) q[4], q[9];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[5];
rxx(0) q[7], q[10];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[4];
rxx(0) q[5], q[0];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[3];
rxx(0) q[2], q[13];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[13];
rxx(0) q[8], q[9];
rxx(0) q[8], q[11];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[11];
rxx(0) q[6], q[4];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
rxx(0) q[10], q[8];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[10];
rxx(0) q[9], q[12];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[10];
rxx(0) q[1], q[0];
rz(1.5707963267948966) q[7];
rxx(0) q[2], q[8];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[10];
rxx(0) q[1], q[10];
rxx(0) q[8], q[9];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
rxx(0) q[6], q[2];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[0];
rxx(0) q[14], q[4];
ry(1.5707963267948966) q[4];
rxx(0) q[14], q[3];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[9];
rxx(0) q[5], q[8];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
rxx(0) q[1], q[5];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
rxx(0) q[0], q[6];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[9];
rxx(0) q[6], q[14];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
rxx(0) q[9], q[0];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
rxx(0) q[6], q[1];
rz(1.5707963267948966) q[3];
rxx(0) q[12], q[5];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
rxx(0) q[13], q[14];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
rxx(0) q[6], q[8];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
rxx(0) q[1], q[14];
rx(1.5707963267948966) q[9];
rxx(0) q[10], q[2];
rxx(0) q[0], q[13];
rxx(0) q[9], q[3];
rz(1.5707963267948966) q[11];
rxx(0) q[13], q[3];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[7];
rxx(0) q[9], q[2];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[13];
rxx(0) q[12], q[13];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[7];
rxx(0) q[7], q[6];
rxx(0) q[1], q[11];
rxx(0) q[12], q[1];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
rxx(0) q[3], q[14];
rxx(0) q[9], q[11];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
rxx(0) q[12], q[0];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[6];
rxx(0) q[2], q[3];
rx(1.5707963267948966) q[5];
rxx(0) q[7], q[5];
rxx(0) q[9], q[13];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
rxx(0) q[14], q[8];
rx(1.5707963267948966) q[2];
rxx(0) q[14], q[6];
rxx(0) q[12], q[14];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
rxx(0) q[9], q[14];
rz(1.5707963267948966) q[13];
rxx(0) q[4], q[1];
rz(1.5707963267948966) q[6];
rxx(0) q[12], q[6];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[7];
rxx(0) q[2], q[0];
ry(1.5707963267948966) q[2];
rxx(0) q[8], q[0];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
rxx(0) q[4], q[5];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
rxx(0) q[11], q[10];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[1];
rxx(0) q[7], q[4];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[1];
rxx(0) q[6], q[1];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[1];
rxx(0) q[0], q[12];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[0];
rxx(0) q[5], q[13];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
rxx(0) q[14], q[11];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
rxx(0) q[2], q[11];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[10];
rxx(0) q[9], q[11];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[3];
rxx(0) q[12], q[14];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rxx(0) q[12], q[9];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[4];
rxx(0) q[8], q[9];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[0];
rxx(0) q[4], q[13];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[7];
rxx(0) q[11], q[2];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
rxx(0) q[6], q[0];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[12];
rxx(0) q[11], q[0];
ry(1.5707963267948966) q[5];
rxx(0) q[3], q[5];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
rxx(0) q[14], q[6];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
rxx(0) q[14], q[10];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[14];
rxx(0) q[7], q[5];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[14];
rxx(0) q[4], q[6];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[12];
rxx(0) q[6], q[12];
rxx(0) q[6], q[14];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
rxx(0) q[11], q[12];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[11];
rxx(0) q[8], q[4];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[12];
rxx(0) q[0], q[11];
ry(1.5707963267948966) q[11];
rxx(0) q[11], q[6];
rz(1.5707963267948966) q[13];
rxx(0) q[13], q[7];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[7];
rxx(0) q[2], q[5];
rz(1.5707963267948966) q[10];
rxx(0) q[10], q[1];
rxx(0) q[14], q[7];
rxx(0) q[6], q[7];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[14];
rxx(0) q[1], q[10];
ry(1.5707963267948966) q[6];
rxx(0) q[1], q[11];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
rxx(0) q[9], q[14];
ry(1.5707963267948966) q[6];
rxx(0) q[8], q[11];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
rxx(0) q[8], q[5];
rx(1.5707963267948966) q[14];
rxx(0) q[1], q[4];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
rxx(0) q[12], q[5];
rxx(0) q[0], q[6];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
rxx(0) q[5], q[6];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[12];
rxx(0) q[9], q[5];
rxx(0) q[6], q[2];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[14];
rxx(0) q[11], q[10];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
rxx(0) q[5], q[1];
ry(1.5707963267948966) q[4];
rxx(0) q[10], q[1];
ry(1.5707963267948966) q[7];
rxx(0) q[4], q[10];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
