OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
ccx q[27], q[26], q[28];
x q[27];
ccx q[25], q[29], q[27];
ccx q[6], q[20], q[29];
ccx q[25], q[29], q[27];
ccx q[6], q[20], q[29];
ccx q[27], q[26], q[28];
x q[12];
x q[14];
x q[18];
x q[26];
ccx q[18], q[29], q[26];
ccx q[12], q[14], q[29];
ccx q[18], q[29], q[26];
ccx q[12], q[14], q[29];
x q[12];
x q[14];
x q[18];
ccx q[27], q[26], q[28];
x q[27];
ccx q[25], q[29], q[27];
ccx q[6], q[20], q[29];
ccx q[25], q[29], q[27];
ccx q[6], q[20], q[29];
ccx q[27], q[26], q[28];
x q[12];
x q[14];
x q[18];
x q[26];
ccx q[18], q[29], q[26];
ccx q[12], q[14], q[29];
ccx q[18], q[29], q[26];
ccx q[12], q[14], q[29];
x q[12];
x q[14];
x q[18];
x q[8];
x q[11];
x q[29];
ccx q[11], q[28], q[29];
ccx q[3], q[8], q[28];
ccx q[11], q[28], q[29];
ccx q[3], q[8], q[28];
x q[8];
x q[11];
ccx q[28], q[29], q[26];
ccx q[27], q[26], q[28];
x q[27];
ccx q[25], q[29], q[27];
ccx q[6], q[20], q[29];
ccx q[25], q[29], q[27];
ccx q[6], q[20], q[29];
ccx q[27], q[26], q[28];
x q[12];
x q[14];
x q[18];
x q[26];
ccx q[18], q[29], q[26];
ccx q[12], q[14], q[29];
ccx q[18], q[29], q[26];
ccx q[12], q[14], q[29];
x q[12];
x q[14];
x q[18];
ccx q[27], q[26], q[28];
x q[27];
ccx q[25], q[29], q[27];
ccx q[6], q[20], q[29];
ccx q[25], q[29], q[27];
ccx q[6], q[20], q[29];
ccx q[27], q[26], q[28];
x q[12];
x q[14];
x q[18];
x q[26];
ccx q[18], q[29], q[26];
ccx q[12], q[14], q[29];
ccx q[18], q[29], q[26];
ccx q[12], q[14], q[29];
x q[12];
x q[14];
x q[18];
x q[8];
x q[11];
x q[29];
ccx q[11], q[28], q[29];
ccx q[3], q[8], q[28];
ccx q[11], q[28], q[29];
ccx q[3], q[8], q[28];
x q[8];
x q[11];