OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
h q[4];
cx q[6], q[1];
h q[9];
h q[11];
cx q[0], q[4];
h q[0];
cx q[1], q[12];
h q[6];
cx q[12], q[5];
h q[3];
cx q[12], q[11];
cx q[11], q[10];
cx q[10], q[3];
h q[7];
h q[6];
cx q[9], q[5];
h q[5];
cx q[1], q[4];
h q[8];
h q[2];
h q[11];
cx q[2], q[0];
cx q[0], q[3];
h q[6];
cx q[8], q[0];
h q[0];
h q[5];
h q[8];
cx q[4], q[8];
h q[9];
cx q[5], q[3];
cx q[7], q[9];
h q[0];
h q[8];
cx q[12], q[11];
cx q[10], q[6];
h q[12];
h q[0];
h q[6];
cx q[11], q[5];
h q[7];
cx q[0], q[8];
h q[2];
cx q[4], q[5];
cx q[11], q[3];
h q[1];
cx q[5], q[11];
h q[2];
cx q[0], q[9];
h q[2];
h q[2];
cx q[7], q[8];
cx q[10], q[12];
h q[7];
cx q[8], q[4];
h q[11];
h q[5];
h q[12];
cx q[5], q[8];
cx q[6], q[4];
h q[3];
cx q[9], q[1];
h q[0];
cx q[5], q[3];
h q[0];
h q[6];
h q[4];
cx q[8], q[11];
cx q[8], q[7];
h q[12];
cx q[7], q[1];
cx q[7], q[12];
cx q[3], q[6];
cx q[6], q[10];
cx q[12], q[9];
cx q[10], q[5];
cx q[7], q[10];
cx q[0], q[7];
cx q[11], q[2];
cx q[3], q[0];
h q[1];
h q[11];
cx q[5], q[10];
h q[0];
h q[12];
cx q[8], q[11];
h q[4];
h q[12];
h q[3];
h q[4];
cx q[8], q[10];
cx q[9], q[6];
cx q[5], q[1];
h q[5];
cx q[6], q[9];
cx q[12], q[8];
h q[7];
cx q[6], q[9];
cx q[8], q[9];
cx q[6], q[2];
h q[4];
cx q[2], q[11];
cx q[7], q[2];
cx q[4], q[3];
h q[0];
cx q[7], q[8];
h q[6];
cx q[0], q[1];
h q[12];
h q[1];
h q[7];
cx q[0], q[7];
cx q[8], q[9];
cx q[7], q[6];
h q[9];
cx q[7], q[3];
h q[0];
cx q[11], q[12];
h q[2];
cx q[11], q[7];
h q[8];
h q[7];
cx q[12], q[9];
cx q[6], q[0];
h q[6];
h q[10];
cx q[12], q[1];
h q[6];
cx q[6], q[9];
cx q[9], q[4];
cx q[8], q[10];
cx q[1], q[2];
h q[3];
h q[2];
cx q[3], q[4];
cx q[3], q[11];
h q[12];
h q[1];
cx q[6], q[2];
cx q[1], q[12];
cx q[1], q[3];
cx q[2], q[12];
h q[11];
cx q[6], q[4];
h q[4];
cx q[0], q[3];
cx q[7], q[9];
h q[9];
cx q[6], q[4];
h q[0];
h q[4];
cx q[8], q[0];
cx q[1], q[10];
cx q[2], q[11];
h q[9];
cx q[8], q[2];
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
