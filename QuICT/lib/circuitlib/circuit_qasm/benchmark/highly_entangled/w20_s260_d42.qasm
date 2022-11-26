OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
h q[11];
h q[4];
cx q[17], q[1];
cx q[2], q[4];
cx q[12], q[0];
cx q[0], q[6];
cx q[7], q[5];
cx q[4], q[18];
cx q[19], q[1];
cx q[15], q[16];
h q[1];
cx q[11], q[18];
h q[6];
cx q[6], q[1];
h q[0];
cx q[0], q[6];
cx q[19], q[0];
h q[17];
h q[16];
cx q[4], q[2];
cx q[9], q[13];
cx q[11], q[5];
cx q[18], q[5];
cx q[16], q[13];
h q[1];
cx q[13], q[9];
cx q[0], q[4];
h q[18];
h q[10];
cx q[17], q[4];
h q[12];
cx q[13], q[9];
cx q[10], q[0];
cx q[11], q[10];
h q[8];
cx q[9], q[6];
h q[18];
h q[16];
h q[4];
h q[10];
h q[18];
cx q[1], q[0];
h q[9];
h q[7];
cx q[9], q[17];
cx q[4], q[16];
cx q[7], q[1];
h q[0];
cx q[13], q[2];
cx q[2], q[18];
h q[4];
h q[5];
h q[13];
cx q[8], q[10];
cx q[19], q[14];
h q[9];
h q[4];
cx q[11], q[3];
h q[18];
h q[19];
h q[16];
h q[11];
h q[11];
cx q[2], q[18];
cx q[12], q[11];
cx q[14], q[11];
h q[3];
cx q[9], q[0];
h q[18];
cx q[8], q[0];
cx q[16], q[0];
h q[6];
cx q[12], q[8];
h q[18];
cx q[7], q[13];
cx q[5], q[7];
h q[4];
cx q[12], q[14];
h q[6];
cx q[17], q[18];
h q[7];
h q[4];
h q[0];
h q[3];
h q[1];
h q[0];
cx q[3], q[7];
h q[0];
h q[8];
h q[4];
h q[18];
h q[16];
h q[5];
cx q[14], q[18];
h q[11];
h q[0];
cx q[8], q[3];
h q[17];
h q[10];
h q[3];
h q[18];
cx q[16], q[10];
cx q[7], q[17];
cx q[15], q[19];
h q[9];
cx q[10], q[1];
h q[6];
cx q[0], q[5];
h q[3];
cx q[8], q[10];
h q[8];
cx q[8], q[18];
cx q[3], q[5];
h q[11];
h q[7];
cx q[17], q[14];
cx q[7], q[10];
h q[8];
cx q[15], q[2];
h q[0];
cx q[1], q[2];
h q[16];
h q[19];
cx q[15], q[11];
h q[13];
cx q[4], q[1];
h q[7];
h q[11];
h q[4];
cx q[15], q[4];
cx q[12], q[3];
cx q[4], q[12];
h q[2];
cx q[12], q[6];
h q[11];
h q[10];
h q[3];
cx q[8], q[4];
cx q[2], q[17];
h q[14];
cx q[11], q[18];
cx q[0], q[15];
cx q[17], q[2];
cx q[14], q[6];
cx q[5], q[0];
cx q[9], q[2];
h q[17];
cx q[0], q[4];
h q[15];
h q[5];
h q[12];
h q[18];
h q[19];
h q[3];
h q[19];
cx q[1], q[14];
h q[1];
cx q[19], q[11];
cx q[6], q[11];
cx q[14], q[5];
h q[4];
h q[7];
cx q[6], q[10];
cx q[19], q[10];
h q[15];
cx q[15], q[0];
cx q[18], q[13];
h q[8];
cx q[10], q[14];
cx q[3], q[18];
cx q[15], q[12];
h q[9];
h q[18];
cx q[10], q[5];
cx q[11], q[8];
h q[7];
cx q[10], q[18];
h q[7];
cx q[8], q[12];
h q[4];
cx q[8], q[5];
cx q[7], q[8];
h q[8];
h q[13];
h q[14];
h q[10];
cx q[1], q[10];
h q[3];
cx q[3], q[2];
h q[18];
cx q[0], q[15];
h q[3];
cx q[2], q[4];
h q[6];
h q[14];
h q[8];
cx q[3], q[10];
h q[8];
h q[5];
h q[19];
h q[16];
cx q[1], q[0];
cx q[3], q[10];
h q[10];
cx q[11], q[6];
h q[17];
cx q[14], q[13];
cx q[4], q[18];
cx q[6], q[12];
h q[19];
h q[14];
h q[12];
cx q[12], q[6];
cx q[10], q[0];
cx q[2], q[0];
cx q[2], q[11];
h q[5];
cx q[5], q[10];
cx q[17], q[5];
h q[4];
h q[14];
cx q[1], q[13];
cx q[8], q[14];
cx q[12], q[8];
h q[0];
h q[12];
h q[17];
h q[17];
h q[10];
cx q[17], q[12];
cx q[12], q[10];
h q[11];
cx q[7], q[2];
h q[8];
cx q[16], q[5];
cx q[8], q[19];
h q[4];
cx q[9], q[7];
cx q[7], q[15];
h q[15];
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
measure q[18] -> c[18];
measure q[19] -> c[19];
