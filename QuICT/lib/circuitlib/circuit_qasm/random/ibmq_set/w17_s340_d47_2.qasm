OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
rz(1.5707963267948966) q[0];
cx q[3], q[15];
cx q[15], q[14];
rz(1.5707963267948966) q[7];
cx q[5], q[7];
cx q[2], q[7];
sx q[4];
x q[11];
sx q[5];
x q[1];
x q[16];
sx q[4];
sx q[3];
cx q[8], q[7];
rz(1.5707963267948966) q[14];
cx q[8], q[10];
rz(1.5707963267948966) q[1];
x q[9];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[9];
sx q[12];
x q[15];
rz(1.5707963267948966) q[4];
cx q[11], q[6];
sx q[10];
cx q[1], q[10];
sx q[0];
sx q[9];
x q[2];
sx q[7];
rz(1.5707963267948966) q[11];
cx q[11], q[16];
sx q[8];
sx q[10];
x q[5];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[10];
sx q[6];
rz(1.5707963267948966) q[13];
x q[12];
sx q[9];
x q[3];
x q[15];
x q[2];
cx q[3], q[6];
x q[5];
sx q[4];
x q[13];
cx q[6], q[13];
x q[5];
sx q[4];
sx q[0];
sx q[5];
cx q[5], q[11];
cx q[2], q[1];
rz(1.5707963267948966) q[15];
cx q[13], q[1];
sx q[0];
sx q[0];
cx q[16], q[12];
cx q[15], q[0];
x q[9];
cx q[12], q[7];
x q[13];
cx q[6], q[10];
cx q[15], q[7];
x q[12];
sx q[7];
sx q[8];
x q[14];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[16];
cx q[5], q[0];
sx q[13];
sx q[0];
cx q[12], q[13];
x q[13];
sx q[1];
cx q[6], q[8];
rz(1.5707963267948966) q[0];
cx q[14], q[10];
cx q[2], q[4];
x q[15];
cx q[13], q[3];
sx q[7];
sx q[10];
rz(1.5707963267948966) q[6];
cx q[8], q[12];
rz(1.5707963267948966) q[8];
x q[0];
rz(1.5707963267948966) q[11];
cx q[5], q[15];
sx q[5];
cx q[14], q[10];
cx q[16], q[15];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[3];
sx q[8];
sx q[8];
x q[7];
cx q[0], q[12];
sx q[16];
rz(1.5707963267948966) q[15];
cx q[5], q[4];
rz(1.5707963267948966) q[2];
sx q[15];
sx q[5];
sx q[3];
x q[10];
rz(1.5707963267948966) q[1];
cx q[7], q[0];
sx q[9];
rz(1.5707963267948966) q[10];
sx q[11];
rz(1.5707963267948966) q[8];
x q[5];
rz(1.5707963267948966) q[15];
sx q[3];
cx q[8], q[11];
x q[15];
x q[16];
rz(1.5707963267948966) q[12];
cx q[8], q[3];
x q[15];
cx q[8], q[15];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[1];
x q[10];
x q[15];
cx q[1], q[15];
sx q[6];
rz(1.5707963267948966) q[2];
x q[4];
x q[7];
rz(1.5707963267948966) q[8];
sx q[16];
cx q[10], q[9];
rz(1.5707963267948966) q[7];
cx q[16], q[10];
cx q[14], q[9];
cx q[1], q[11];
sx q[5];
x q[14];
x q[10];
rz(1.5707963267948966) q[5];
x q[11];
sx q[10];
rz(1.5707963267948966) q[6];
x q[13];
x q[5];
sx q[1];
rz(1.5707963267948966) q[12];
x q[13];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[11];
x q[10];
sx q[3];
cx q[15], q[12];
x q[4];
cx q[10], q[15];
x q[15];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
sx q[12];
cx q[14], q[3];
x q[9];
cx q[5], q[2];
x q[2];
sx q[12];
x q[12];
x q[11];
sx q[15];
rz(1.5707963267948966) q[8];
cx q[16], q[5];
cx q[14], q[11];
sx q[1];
rz(1.5707963267948966) q[0];
x q[8];
rz(1.5707963267948966) q[13];
x q[8];
cx q[3], q[14];
sx q[0];
x q[14];
cx q[12], q[16];
x q[0];
cx q[9], q[4];
cx q[1], q[4];
x q[3];
rz(1.5707963267948966) q[6];
sx q[0];
cx q[4], q[10];
x q[4];
sx q[14];
x q[6];
cx q[0], q[9];
cx q[13], q[6];
rz(1.5707963267948966) q[13];
sx q[11];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[9];
x q[7];
x q[1];
rz(1.5707963267948966) q[7];
sx q[5];
sx q[15];
sx q[4];
x q[15];
sx q[3];
sx q[8];
cx q[16], q[1];
sx q[8];
cx q[0], q[16];
sx q[14];
sx q[5];
x q[1];
rz(1.5707963267948966) q[0];
x q[11];
rz(1.5707963267948966) q[1];
x q[14];
x q[11];
rz(1.5707963267948966) q[12];
sx q[7];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[10];
x q[5];
rz(1.5707963267948966) q[2];
cx q[5], q[15];
x q[12];
rz(1.5707963267948966) q[16];
sx q[6];
rz(1.5707963267948966) q[14];
sx q[2];
rz(1.5707963267948966) q[5];
x q[3];
rz(1.5707963267948966) q[16];
x q[1];
sx q[16];
sx q[4];
sx q[1];
rz(1.5707963267948966) q[6];
cx q[12], q[14];
x q[11];
sx q[0];
rz(1.5707963267948966) q[5];
x q[14];
cx q[16], q[6];
cx q[4], q[10];
sx q[0];
sx q[7];
rz(1.5707963267948966) q[13];
x q[16];
rz(1.5707963267948966) q[3];
x q[9];
sx q[0];
rz(1.5707963267948966) q[16];
sx q[11];
rz(1.5707963267948966) q[0];
cx q[11], q[4];
cx q[7], q[10];
x q[10];
cx q[7], q[12];
cx q[6], q[15];
x q[9];
sx q[0];
sx q[6];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[4];
x q[6];
x q[2];
x q[10];
sx q[13];
sx q[8];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[5];
sx q[0];
sx q[4];
sx q[0];
sx q[15];
sx q[7];
cx q[12], q[11];
x q[8];
cx q[10], q[8];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[10];
x q[8];
cx q[2], q[9];
cx q[11], q[10];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[9];
cx q[15], q[13];
cx q[7], q[12];
cx q[16], q[10];
sx q[13];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[9];
x q[5];
cx q[2], q[4];
rz(1.5707963267948966) q[4];
x q[3];
x q[12];
rz(1.5707963267948966) q[10];
x q[1];
cx q[10], q[15];
x q[3];
rz(1.5707963267948966) q[13];
cx q[8], q[13];
cx q[8], q[1];
cx q[3], q[13];
rz(1.5707963267948966) q[5];
cx q[1], q[7];
cx q[11], q[8];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[8];
sx q[8];
sx q[13];
x q[5];
rz(1.5707963267948966) q[14];
cx q[5], q[13];
sx q[10];
sx q[2];
cx q[4], q[9];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[4];
sx q[2];
x q[14];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[13];
x q[7];
x q[8];
rz(1.5707963267948966) q[2];
sx q[14];
sx q[4];
cx q[5], q[10];
sx q[2];
