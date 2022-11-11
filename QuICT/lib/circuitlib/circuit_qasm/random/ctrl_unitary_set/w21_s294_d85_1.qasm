OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
cu3(1.5707963267948966, 0, 0) q[8], q[19];
cu3(1.5707963267948966, 0, 0) q[4], q[9];
ch q[19], q[1];
cy q[16], q[13];
cy q[8], q[7];
cy q[14], q[2];
cy q[6], q[16];
cy q[18], q[6];
cx q[17], q[18];
cx q[1], q[17];
cy q[18], q[13];
cy q[6], q[11];
cx q[19], q[17];
cy q[6], q[18];
ch q[4], q[16];
cy q[2], q[1];
cy q[10], q[6];
ch q[12], q[20];
cy q[5], q[1];
cu3(1.5707963267948966, 0, 0) q[11], q[1];
cy q[20], q[19];
cy q[5], q[4];
cu3(1.5707963267948966, 0, 0) q[17], q[13];
cu3(1.5707963267948966, 0, 0) q[14], q[15];
ch q[17], q[3];
cx q[13], q[4];
ch q[10], q[3];
cu3(1.5707963267948966, 0, 0) q[19], q[1];
cy q[15], q[9];
cx q[0], q[16];
cu3(1.5707963267948966, 0, 0) q[18], q[17];
cu3(1.5707963267948966, 0, 0) q[18], q[2];
ch q[4], q[9];
cu3(1.5707963267948966, 0, 0) q[8], q[19];
cy q[17], q[13];
cu3(1.5707963267948966, 0, 0) q[11], q[15];
ch q[17], q[7];
cu3(1.5707963267948966, 0, 0) q[14], q[5];
cu3(1.5707963267948966, 0, 0) q[20], q[9];
cy q[20], q[3];
cu3(1.5707963267948966, 0, 0) q[10], q[15];
ch q[2], q[9];
cx q[3], q[8];
cy q[10], q[20];
ch q[9], q[10];
cx q[8], q[5];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
cu3(1.5707963267948966, 0, 0) q[18], q[9];
ch q[13], q[10];
ch q[11], q[15];
cu3(1.5707963267948966, 0, 0) q[6], q[8];
ch q[0], q[20];
cy q[13], q[19];
cx q[20], q[5];
cx q[17], q[6];
ch q[12], q[2];
cx q[8], q[14];
cy q[1], q[3];
cu3(1.5707963267948966, 0, 0) q[17], q[3];
cy q[8], q[9];
cy q[14], q[3];
ch q[13], q[18];
cu3(1.5707963267948966, 0, 0) q[1], q[18];
cy q[8], q[17];
cu3(1.5707963267948966, 0, 0) q[16], q[4];
cx q[1], q[2];
cx q[20], q[18];
cu3(1.5707963267948966, 0, 0) q[15], q[6];
cx q[11], q[6];
cx q[15], q[8];
cx q[6], q[11];
cy q[6], q[11];
cu3(1.5707963267948966, 0, 0) q[12], q[20];
ch q[5], q[12];
cy q[6], q[1];
ch q[6], q[15];
ch q[8], q[19];
cu3(1.5707963267948966, 0, 0) q[7], q[18];
cy q[3], q[20];
cy q[12], q[18];
cy q[17], q[13];
cy q[6], q[14];
cy q[12], q[9];
cx q[4], q[12];
ch q[18], q[17];
ch q[9], q[11];
cy q[6], q[1];
cy q[13], q[15];
cy q[3], q[7];
cu3(1.5707963267948966, 0, 0) q[7], q[5];
ch q[19], q[7];
cx q[2], q[9];
cu3(1.5707963267948966, 0, 0) q[4], q[15];
cy q[9], q[3];
cu3(1.5707963267948966, 0, 0) q[14], q[15];
cx q[12], q[10];
cx q[6], q[4];
cu3(1.5707963267948966, 0, 0) q[20], q[8];
ch q[3], q[14];
cu3(1.5707963267948966, 0, 0) q[20], q[14];
cy q[3], q[13];
ch q[7], q[15];
cy q[8], q[14];
cu3(1.5707963267948966, 0, 0) q[16], q[4];
cy q[8], q[15];
cu3(1.5707963267948966, 0, 0) q[10], q[11];
cy q[19], q[16];
cx q[15], q[11];
ch q[12], q[8];
ch q[5], q[18];
ch q[17], q[11];
cy q[12], q[2];
cu3(1.5707963267948966, 0, 0) q[20], q[7];
cy q[7], q[2];
ch q[15], q[13];
cu3(1.5707963267948966, 0, 0) q[16], q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[14];
cx q[19], q[6];
cu3(1.5707963267948966, 0, 0) q[1], q[8];
ch q[20], q[19];
cx q[5], q[9];
ch q[17], q[5];
cu3(1.5707963267948966, 0, 0) q[3], q[9];
cx q[15], q[10];
ch q[5], q[0];
cy q[13], q[20];
cy q[8], q[9];
cy q[11], q[18];
cy q[3], q[7];
ch q[16], q[6];
cy q[19], q[0];
cy q[16], q[19];
cx q[4], q[11];
cu3(1.5707963267948966, 0, 0) q[14], q[6];
cy q[16], q[4];
cu3(1.5707963267948966, 0, 0) q[11], q[7];
cx q[16], q[1];
cx q[2], q[12];
ch q[7], q[3];
cx q[8], q[10];
cx q[0], q[4];
cy q[7], q[11];
cu3(1.5707963267948966, 0, 0) q[17], q[14];
ch q[11], q[2];
ch q[18], q[12];
cy q[15], q[2];
cx q[10], q[1];
cy q[13], q[15];
cu3(1.5707963267948966, 0, 0) q[4], q[9];
ch q[13], q[7];
cu3(1.5707963267948966, 0, 0) q[15], q[8];
cu3(1.5707963267948966, 0, 0) q[2], q[18];
ch q[7], q[20];
cx q[19], q[10];
cy q[7], q[2];
cu3(1.5707963267948966, 0, 0) q[5], q[4];
cu3(1.5707963267948966, 0, 0) q[7], q[15];
cu3(1.5707963267948966, 0, 0) q[15], q[2];
cu3(1.5707963267948966, 0, 0) q[13], q[2];
ch q[12], q[0];
ch q[6], q[2];
ch q[6], q[3];
cx q[3], q[16];
cu3(1.5707963267948966, 0, 0) q[19], q[17];
ch q[17], q[13];
cy q[4], q[18];
cu3(1.5707963267948966, 0, 0) q[8], q[10];
cx q[5], q[18];
cy q[14], q[5];
ch q[1], q[11];
cx q[18], q[17];
cu3(1.5707963267948966, 0, 0) q[2], q[18];
cx q[1], q[3];
cy q[10], q[16];
ch q[9], q[15];
cu3(1.5707963267948966, 0, 0) q[9], q[12];
cx q[9], q[8];
cu3(1.5707963267948966, 0, 0) q[4], q[15];
cx q[5], q[19];
ch q[18], q[13];
ch q[17], q[3];
ch q[17], q[3];
ch q[1], q[12];
ch q[8], q[13];
cy q[20], q[17];
cy q[5], q[9];
ch q[5], q[14];
ch q[17], q[1];
cy q[6], q[9];
ch q[14], q[11];
cu3(1.5707963267948966, 0, 0) q[14], q[6];
cx q[17], q[15];
cu3(1.5707963267948966, 0, 0) q[5], q[12];
cy q[4], q[19];
ch q[18], q[10];
cx q[1], q[16];
cx q[10], q[13];
ch q[9], q[1];
cy q[18], q[9];
cu3(1.5707963267948966, 0, 0) q[18], q[9];
cx q[6], q[20];
ch q[19], q[16];
ch q[10], q[18];
cx q[3], q[10];
cx q[19], q[8];
cu3(1.5707963267948966, 0, 0) q[14], q[9];
cx q[3], q[17];
ch q[4], q[18];
cy q[10], q[20];
cy q[6], q[0];
ch q[3], q[12];
cu3(1.5707963267948966, 0, 0) q[13], q[15];
cy q[4], q[6];
cy q[0], q[3];
cu3(1.5707963267948966, 0, 0) q[7], q[19];
ch q[0], q[20];
cy q[1], q[15];
cy q[15], q[10];
cy q[15], q[12];
cu3(1.5707963267948966, 0, 0) q[11], q[10];
ch q[1], q[16];
cu3(1.5707963267948966, 0, 0) q[8], q[9];
cy q[6], q[13];
ch q[11], q[18];
cu3(1.5707963267948966, 0, 0) q[6], q[3];
cu3(1.5707963267948966, 0, 0) q[19], q[5];
cu3(1.5707963267948966, 0, 0) q[8], q[18];
cx q[9], q[19];
ch q[16], q[2];
ch q[12], q[10];
ch q[17], q[12];
cx q[19], q[6];
cx q[15], q[18];
ch q[12], q[9];
cy q[9], q[16];
cy q[0], q[13];
cy q[12], q[10];
cy q[17], q[13];
cy q[10], q[18];
cu3(1.5707963267948966, 0, 0) q[5], q[11];
cx q[12], q[9];
ch q[13], q[7];
cu3(1.5707963267948966, 0, 0) q[18], q[4];
cy q[16], q[9];
ch q[9], q[20];
cu3(1.5707963267948966, 0, 0) q[3], q[20];
ch q[17], q[3];
ch q[13], q[7];
cu3(1.5707963267948966, 0, 0) q[18], q[14];
cy q[3], q[8];
cx q[1], q[18];
cy q[5], q[20];
ch q[10], q[13];
ch q[5], q[10];
cy q[6], q[10];
cx q[4], q[6];
cy q[9], q[10];
ch q[9], q[0];
cy q[8], q[17];
ch q[6], q[19];
cx q[8], q[0];
cu3(1.5707963267948966, 0, 0) q[7], q[14];
cx q[20], q[9];
ch q[2], q[15];
cy q[3], q[4];
ch q[19], q[14];
cx q[6], q[9];
cx q[19], q[5];
cy q[18], q[14];
cu3(1.5707963267948966, 0, 0) q[13], q[2];
cu3(1.5707963267948966, 0, 0) q[5], q[8];
cx q[4], q[5];
cy q[7], q[20];
ch q[13], q[3];
cy q[0], q[8];
cx q[1], q[14];
cy q[19], q[5];
cu3(1.5707963267948966, 0, 0) q[10], q[16];
ch q[19], q[11];
ch q[13], q[9];
cu3(1.5707963267948966, 0, 0) q[6], q[5];
cu3(1.5707963267948966, 0, 0) q[0], q[3];
cx q[5], q[15];
cx q[15], q[14];
cx q[5], q[6];
cx q[11], q[19];
ch q[7], q[11];
ch q[0], q[15];
cy q[18], q[3];
cu3(1.5707963267948966, 0, 0) q[7], q[11];
ch q[14], q[5];
cu3(1.5707963267948966, 0, 0) q[2], q[16];
cy q[1], q[9];
cx q[6], q[13];
