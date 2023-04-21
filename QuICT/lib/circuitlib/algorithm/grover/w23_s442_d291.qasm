OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[21];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
x q[21];
h q[21];
x q[9];
x q[10];
x q[11];
x q[13];
x q[14];
x q[16];
x q[17];
x q[18];
x q[20];
ccx q[10], q[21], q[22];
ccx q[9], q[20], q[21];
ccx q[8], q[19], q[20];
ccx q[7], q[18], q[19];
ccx q[6], q[17], q[18];
ccx q[5], q[16], q[17];
ccx q[4], q[15], q[16];
ccx q[3], q[14], q[15];
ccx q[2], q[13], q[14];
ccx q[0], q[1], q[13];
ccx q[2], q[13], q[14];
ccx q[3], q[14], q[15];
ccx q[4], q[15], q[16];
ccx q[5], q[16], q[17];
ccx q[6], q[17], q[18];
ccx q[7], q[18], q[19];
ccx q[8], q[19], q[20];
ccx q[9], q[20], q[21];
ccx q[10], q[21], q[22];
ccx q[9], q[20], q[21];
ccx q[8], q[19], q[20];
ccx q[7], q[18], q[19];
ccx q[6], q[17], q[18];
ccx q[5], q[16], q[17];
ccx q[4], q[15], q[16];
ccx q[3], q[14], q[15];
ccx q[2], q[13], q[14];
ccx q[0], q[1], q[13];
ccx q[2], q[13], q[14];
ccx q[3], q[14], q[15];
ccx q[4], q[15], q[16];
ccx q[5], q[16], q[17];
ccx q[6], q[17], q[18];
ccx q[7], q[18], q[19];
ccx q[8], q[19], q[20];
ccx q[9], q[20], q[21];
h q[21];
s q[22];
ccx q[21], q[10], q[22];
ccx q[20], q[9], q[10];
ccx q[19], q[8], q[9];
ccx q[18], q[7], q[8];
ccx q[17], q[6], q[7];
ccx q[16], q[5], q[6];
ccx q[15], q[4], q[5];
ccx q[14], q[3], q[4];
ccx q[13], q[2], q[3];
ccx q[11], q[12], q[2];
ccx q[13], q[2], q[3];
ccx q[14], q[3], q[4];
ccx q[15], q[4], q[5];
ccx q[16], q[5], q[6];
ccx q[17], q[6], q[7];
ccx q[18], q[7], q[8];
ccx q[19], q[8], q[9];
ccx q[20], q[9], q[10];
ccx q[21], q[10], q[22];
ccx q[20], q[9], q[10];
ccx q[19], q[8], q[9];
ccx q[18], q[7], q[8];
ccx q[17], q[6], q[7];
ccx q[16], q[5], q[6];
ccx q[15], q[4], q[5];
ccx q[14], q[3], q[4];
ccx q[13], q[2], q[3];
ccx q[11], q[12], q[2];
ccx q[13], q[2], q[3];
ccx q[14], q[3], q[4];
ccx q[15], q[4], q[5];
ccx q[16], q[5], q[6];
ccx q[17], q[6], q[7];
ccx q[18], q[7], q[8];
ccx q[19], q[8], q[9];
ccx q[20], q[9], q[10];
sdg q[22];
ccx q[10], q[21], q[22];
ccx q[9], q[20], q[21];
ccx q[8], q[19], q[20];
ccx q[7], q[18], q[19];
ccx q[6], q[17], q[18];
ccx q[5], q[16], q[17];
ccx q[4], q[15], q[16];
ccx q[3], q[14], q[15];
ccx q[2], q[13], q[14];
ccx q[0], q[1], q[13];
ccx q[2], q[13], q[14];
ccx q[3], q[14], q[15];
ccx q[4], q[15], q[16];
ccx q[5], q[16], q[17];
ccx q[6], q[17], q[18];
ccx q[7], q[18], q[19];
ccx q[8], q[19], q[20];
ccx q[9], q[20], q[21];
ccx q[10], q[21], q[22];
ccx q[9], q[20], q[21];
ccx q[8], q[19], q[20];
ccx q[7], q[18], q[19];
ccx q[6], q[17], q[18];
ccx q[5], q[16], q[17];
ccx q[4], q[15], q[16];
ccx q[3], q[14], q[15];
ccx q[2], q[13], q[14];
ccx q[0], q[1], q[13];
ccx q[2], q[13], q[14];
ccx q[3], q[14], q[15];
ccx q[4], q[15], q[16];
ccx q[5], q[16], q[17];
ccx q[6], q[17], q[18];
ccx q[7], q[18], q[19];
ccx q[8], q[19], q[20];
ccx q[9], q[20], q[21];
s q[22];
ccx q[21], q[10], q[22];
ccx q[20], q[9], q[10];
ccx q[19], q[8], q[9];
ccx q[18], q[7], q[8];
ccx q[17], q[6], q[7];
ccx q[16], q[5], q[6];
ccx q[15], q[4], q[5];
ccx q[14], q[3], q[4];
ccx q[13], q[2], q[3];
ccx q[11], q[12], q[2];
ccx q[13], q[2], q[3];
ccx q[14], q[3], q[4];
ccx q[15], q[4], q[5];
ccx q[16], q[5], q[6];
ccx q[17], q[6], q[7];
ccx q[18], q[7], q[8];
ccx q[19], q[8], q[9];
ccx q[20], q[9], q[10];
ccx q[21], q[10], q[22];
ccx q[20], q[9], q[10];
ccx q[19], q[8], q[9];
ccx q[18], q[7], q[8];
ccx q[17], q[6], q[7];
ccx q[16], q[5], q[6];
ccx q[15], q[4], q[5];
ccx q[14], q[3], q[4];
ccx q[13], q[2], q[3];
ccx q[11], q[12], q[2];
ccx q[13], q[2], q[3];
ccx q[14], q[3], q[4];
ccx q[15], q[4], q[5];
ccx q[16], q[5], q[6];
ccx q[17], q[6], q[7];
ccx q[18], q[7], q[8];
ccx q[19], q[8], q[9];
ccx q[20], q[9], q[10];
h q[21];
sdg q[22];
x q[9];
x q[10];
x q[11];
x q[13];
x q[14];
x q[16];
x q[17];
x q[18];
x q[20];
h q[21];
x q[21];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
x q[6];
x q[7];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[13];
x q[14];
x q[15];
x q[16];
x q[17];
x q[18];
x q[19];
x q[20];
h q[20];
ccx q[10], q[20], q[21];
ccx q[9], q[19], q[20];
ccx q[8], q[18], q[19];
ccx q[7], q[17], q[18];
ccx q[6], q[16], q[17];
ccx q[5], q[15], q[16];
ccx q[4], q[14], q[15];
ccx q[3], q[13], q[14];
ccx q[2], q[12], q[13];
ccx q[0], q[1], q[12];
ccx q[2], q[12], q[13];
ccx q[3], q[13], q[14];
ccx q[4], q[14], q[15];
ccx q[5], q[15], q[16];
ccx q[6], q[16], q[17];
ccx q[7], q[17], q[18];
ccx q[8], q[18], q[19];
ccx q[9], q[19], q[20];
ccx q[10], q[20], q[21];
ccx q[9], q[19], q[20];
ccx q[8], q[18], q[19];
ccx q[7], q[17], q[18];
ccx q[6], q[16], q[17];
ccx q[5], q[15], q[16];
ccx q[4], q[14], q[15];
ccx q[3], q[13], q[14];
ccx q[2], q[12], q[13];
ccx q[0], q[1], q[12];
ccx q[2], q[12], q[13];
ccx q[3], q[13], q[14];
ccx q[4], q[14], q[15];
ccx q[5], q[15], q[16];
ccx q[6], q[16], q[17];
ccx q[7], q[17], q[18];
ccx q[8], q[18], q[19];
ccx q[9], q[19], q[20];
h q[20];
s q[21];
ccx q[20], q[10], q[21];
ccx q[19], q[9], q[10];
ccx q[18], q[8], q[9];
ccx q[17], q[7], q[8];
ccx q[16], q[6], q[7];
ccx q[15], q[5], q[6];
ccx q[14], q[4], q[5];
ccx q[13], q[3], q[4];
ccx q[11], q[12], q[3];
ccx q[13], q[3], q[4];
ccx q[14], q[4], q[5];
ccx q[15], q[5], q[6];
ccx q[16], q[6], q[7];
ccx q[17], q[7], q[8];
ccx q[18], q[8], q[9];
ccx q[19], q[9], q[10];
ccx q[20], q[10], q[21];
ccx q[19], q[9], q[10];
ccx q[18], q[8], q[9];
ccx q[17], q[7], q[8];
ccx q[16], q[6], q[7];
ccx q[15], q[5], q[6];
ccx q[14], q[4], q[5];
ccx q[13], q[3], q[4];
ccx q[11], q[12], q[3];
ccx q[13], q[3], q[4];
ccx q[14], q[4], q[5];
ccx q[15], q[5], q[6];
ccx q[16], q[6], q[7];
ccx q[17], q[7], q[8];
ccx q[18], q[8], q[9];
ccx q[19], q[9], q[10];
sdg q[21];
ccx q[10], q[20], q[21];
ccx q[9], q[19], q[20];
ccx q[8], q[18], q[19];
ccx q[7], q[17], q[18];
ccx q[6], q[16], q[17];
ccx q[5], q[15], q[16];
ccx q[4], q[14], q[15];
ccx q[3], q[13], q[14];
ccx q[2], q[12], q[13];
ccx q[0], q[1], q[12];
ccx q[2], q[12], q[13];
ccx q[3], q[13], q[14];
ccx q[4], q[14], q[15];
ccx q[5], q[15], q[16];
ccx q[6], q[16], q[17];
ccx q[7], q[17], q[18];
ccx q[8], q[18], q[19];
ccx q[9], q[19], q[20];
ccx q[10], q[20], q[21];
ccx q[9], q[19], q[20];
ccx q[8], q[18], q[19];
ccx q[7], q[17], q[18];
ccx q[6], q[16], q[17];
ccx q[5], q[15], q[16];
ccx q[4], q[14], q[15];
ccx q[3], q[13], q[14];
ccx q[2], q[12], q[13];
ccx q[0], q[1], q[12];
ccx q[2], q[12], q[13];
ccx q[3], q[13], q[14];
ccx q[4], q[14], q[15];
ccx q[5], q[15], q[16];
ccx q[6], q[16], q[17];
ccx q[7], q[17], q[18];
ccx q[8], q[18], q[19];
ccx q[9], q[19], q[20];
s q[21];
ccx q[20], q[10], q[21];
ccx q[19], q[9], q[10];
ccx q[18], q[8], q[9];
ccx q[17], q[7], q[8];
ccx q[16], q[6], q[7];
ccx q[15], q[5], q[6];
ccx q[14], q[4], q[5];
ccx q[13], q[3], q[4];
ccx q[11], q[12], q[3];
ccx q[13], q[3], q[4];
ccx q[14], q[4], q[5];
ccx q[15], q[5], q[6];
ccx q[16], q[6], q[7];
ccx q[17], q[7], q[8];
ccx q[18], q[8], q[9];
ccx q[19], q[9], q[10];
ccx q[20], q[10], q[21];
ccx q[19], q[9], q[10];
ccx q[18], q[8], q[9];
ccx q[17], q[7], q[8];
ccx q[16], q[6], q[7];
ccx q[15], q[5], q[6];
ccx q[14], q[4], q[5];
ccx q[13], q[3], q[4];
ccx q[11], q[12], q[3];
ccx q[13], q[3], q[4];
ccx q[14], q[4], q[5];
ccx q[15], q[5], q[6];
ccx q[16], q[6], q[7];
ccx q[17], q[7], q[8];
ccx q[18], q[8], q[9];
ccx q[19], q[9], q[10];
h q[20];
sdg q[21];
h q[20];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
x q[6];
x q[7];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[13];
x q[14];
x q[15];
x q[16];
x q[17];
x q[18];
x q[19];
x q[20];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
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
measure q[20] -> c[20];
