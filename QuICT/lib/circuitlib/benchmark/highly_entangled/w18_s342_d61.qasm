OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
h q[3];
h q[14];
h q[11];
h q[10];
cx q[17], q[1];
h q[4];
h q[17];
cx q[9], q[7];
cx q[4], q[8];
cx q[3], q[4];
h q[0];
h q[13];
h q[16];
h q[3];
cx q[13], q[4];
cx q[17], q[9];
cx q[8], q[15];
cx q[0], q[6];
cx q[8], q[11];
cx q[4], q[9];
h q[11];
cx q[3], q[6];
h q[3];
h q[0];
cx q[11], q[10];
cx q[10], q[5];
h q[8];
cx q[17], q[11];
cx q[17], q[15];
cx q[3], q[1];
h q[6];
h q[14];
h q[15];
h q[15];
cx q[3], q[6];
h q[9];
h q[6];
cx q[10], q[7];
h q[16];
cx q[12], q[16];
cx q[9], q[13];
h q[10];
h q[14];
h q[11];
h q[2];
cx q[5], q[2];
h q[7];
cx q[9], q[6];
cx q[6], q[3];
h q[5];
h q[0];
h q[10];
cx q[11], q[16];
h q[2];
h q[11];
h q[9];
h q[14];
h q[6];
h q[2];
h q[1];
h q[10];
cx q[15], q[7];
h q[2];
cx q[7], q[9];
cx q[17], q[2];
h q[6];
cx q[17], q[4];
h q[13];
h q[8];
cx q[0], q[6];
cx q[16], q[9];
h q[8];
h q[8];
h q[2];
cx q[13], q[10];
cx q[17], q[7];
h q[3];
cx q[13], q[4];
h q[2];
cx q[6], q[15];
h q[9];
cx q[4], q[16];
cx q[11], q[8];
cx q[17], q[13];
h q[2];
h q[2];
cx q[15], q[3];
h q[1];
cx q[7], q[5];
h q[3];
h q[6];
h q[15];
cx q[9], q[17];
cx q[3], q[1];
cx q[8], q[2];
h q[4];
cx q[4], q[16];
cx q[9], q[17];
cx q[12], q[10];
cx q[16], q[11];
cx q[17], q[3];
h q[1];
cx q[13], q[16];
h q[8];
h q[16];
cx q[10], q[17];
h q[9];
cx q[3], q[12];
cx q[15], q[13];
cx q[3], q[9];
h q[10];
h q[6];
cx q[6], q[11];
h q[5];
cx q[6], q[9];
cx q[8], q[1];
h q[0];
h q[13];
cx q[0], q[3];
cx q[0], q[12];
cx q[3], q[16];
cx q[1], q[14];
h q[2];
h q[11];
cx q[13], q[3];
cx q[17], q[4];
cx q[1], q[0];
cx q[11], q[13];
cx q[17], q[16];
h q[14];
h q[3];
cx q[3], q[17];
cx q[10], q[17];
cx q[9], q[1];
h q[14];
h q[17];
h q[0];
cx q[7], q[16];
cx q[5], q[8];
cx q[2], q[1];
h q[17];
h q[3];
cx q[1], q[13];
h q[4];
h q[1];
cx q[1], q[6];
cx q[15], q[16];
cx q[5], q[13];
cx q[1], q[17];
h q[4];
h q[12];
h q[1];
h q[8];
h q[2];
h q[4];
cx q[0], q[13];
h q[0];
cx q[2], q[15];
h q[15];
cx q[13], q[11];
cx q[17], q[5];
cx q[10], q[7];
cx q[3], q[7];
cx q[0], q[5];
cx q[13], q[16];
h q[11];
cx q[17], q[4];
cx q[6], q[11];
h q[17];
cx q[5], q[17];
cx q[6], q[0];
cx q[6], q[17];
cx q[12], q[15];
cx q[8], q[14];
cx q[9], q[0];
h q[0];
cx q[6], q[3];
cx q[17], q[0];
cx q[3], q[1];
cx q[13], q[4];
cx q[9], q[11];
cx q[2], q[8];
cx q[9], q[8];
cx q[1], q[5];
h q[12];
cx q[3], q[9];
cx q[15], q[7];
h q[7];
h q[12];
cx q[11], q[3];
h q[9];
cx q[5], q[2];
cx q[6], q[4];
h q[12];
cx q[7], q[10];
h q[6];
cx q[13], q[8];
cx q[1], q[2];
cx q[6], q[13];
h q[9];
h q[11];
cx q[1], q[2];
cx q[1], q[10];
h q[16];
h q[14];
h q[4];
h q[4];
h q[2];
cx q[5], q[1];
h q[6];
h q[5];
cx q[4], q[3];
h q[1];
h q[14];
cx q[2], q[10];
h q[7];
h q[9];
h q[13];
h q[5];
h q[8];
h q[7];
h q[16];
cx q[14], q[2];
h q[5];
h q[8];
h q[11];
h q[6];
cx q[12], q[2];
cx q[10], q[15];
h q[17];
cx q[5], q[15];
h q[17];
h q[6];
h q[14];
cx q[7], q[14];
h q[7];
cx q[8], q[6];
h q[8];
cx q[5], q[0];
cx q[8], q[3];
cx q[0], q[1];
cx q[2], q[11];
h q[4];
h q[1];
cx q[12], q[15];
h q[9];
cx q[11], q[6];
cx q[4], q[11];
h q[10];
cx q[17], q[7];
h q[6];
h q[2];
cx q[4], q[12];
h q[17];
h q[17];
cx q[6], q[16];
cx q[14], q[13];
cx q[4], q[12];
cx q[9], q[3];
h q[15];
h q[13];
h q[9];
cx q[16], q[7];
h q[2];
h q[6];
h q[5];
h q[2];
cx q[13], q[0];
h q[13];
h q[13];
cx q[11], q[0];
h q[7];
h q[11];
cx q[16], q[4];
h q[0];
h q[2];
h q[13];
cx q[7], q[14];
cx q[9], q[10];
h q[13];
h q[0];
h q[10];
h q[10];
h q[4];
h q[5];
cx q[10], q[8];
h q[14];
h q[11];
h q[2];
cx q[14], q[13];
cx q[7], q[12];
h q[17];
cx q[16], q[10];
h q[12];
cx q[14], q[3];
h q[13];
h q[5];
h q[3];
cx q[15], q[11];
cx q[9], q[14];
h q[2];
h q[9];
h q[13];
h q[6];
cx q[14], q[4];
cx q[12], q[17];
cx q[1], q[7];
h q[7];
h q[3];
h q[16];
cx q[17], q[0];
h q[11];
h q[0];
h q[16];
h q[6];
cx q[6], q[2];
cx q[3], q[13];
cx q[4], q[12];
h q[11];
h q[4];
cx q[12], q[16];
cx q[1], q[17];
h q[8];
cx q[14], q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
